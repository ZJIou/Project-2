import queue
import threading
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from typing import Callable, Optional

# Опциональное шумоподавление
try:
    import noisereduce as nr

    _NR_AVAILABLE = True
except ImportError:
    _NR_AVAILABLE = False

# ──────────────────────────────────────────────
# Конфигурация по умолчанию
# ──────────────────────────────────────────────
DEFAULT_MODEL_SIZE = "small"
DEFAULT_LANGUAGE = "ru"
DEFAULT_DEVICE = "cpu"  # auto | cpu | cuda
DEFAULT_COMPUTE_TYPE = "int8"  # auto | int8 | float16 | float32

SAMPLE_RATE = 16_000
BLOCK_DURATION_SEC = 0.3

# ── Параметры VAD / буферизации ────────────────
SILENCE_THRESHOLD = 0.008  # RMS порог тишины
SILENCE_DURATION = 1.2  # пауза → конец фразы (сек)
MIN_PHRASE_DURATION = 1.5  # не отправлять фрагменты короче (сек)
MAX_PHRASE_DURATION = 20.0  # принудительный сброс буфера (сек)
MIN_TEXT_TOKENS = 3  # игнорировать результаты короче N слов

# ── Параметры Whisper ──────────────────────────
BEAM_SIZE = 5
TEMPERATURE = [0.0, 0.2, 0.4]  # Whisper пробует по очереди при неуверенности
NO_SPEECH_THRESHOLD = 0.6
LOG_PROB_THRESHOLD = -1.0
COMPRESSION_RATIO = 2.4
CONTEXT_WINDOW_CHARS = 300  # символов предыдущего текста в prompt


class RealtimeTranscriber:
    def __init__(
            self,
            on_transcript: Callable[[str], None],
            model_size: str = DEFAULT_MODEL_SIZE,
            language: Optional[str] = DEFAULT_LANGUAGE,
            initial_prompt: Optional[str] = None,
            device: str = DEFAULT_DEVICE,
            compute_type: str = DEFAULT_COMPUTE_TYPE,
            silence_threshold: float = SILENCE_THRESHOLD,
            noise_reduce: bool = True,
    ) -> None:
        self.on_transcript = on_transcript
        self.language = language
        self.initial_prompt = initial_prompt
        self.silence_threshold = silence_threshold
        self.noise_reduce = noise_reduce and _NR_AVAILABLE

        if noise_reduce and not _NR_AVAILABLE:
            print("[Whisper] noisereduce не установлен. Запустите: pip install noisereduce")

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._prev_text = ""

        self._model = self._load_model(model_size, device, compute_type)

    # ── Загрузка модели ────────────────────────

    @staticmethod
    def _load_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
        def _best_compute(dev: str) -> str:
            return "float16" if dev == "cuda" else "int8"

        def _try_load(dev: str, ct: str) -> WhisperModel:
            resolved_ct = _best_compute(dev) if ct == "auto" else ct
            print(f"[Whisper] Загрузка модели «{model_size}» ({dev}/{resolved_ct})…")
            model = WhisperModel(model_size, device=dev, compute_type=resolved_ct)
            print(f"[Whisper] Модель загружена ✓  ({dev}/{resolved_ct})")
            return model

        if device == "auto":
            try:
                return _try_load("cuda", compute_type)
            except Exception as e:
                print(f"[Whisper] CUDA недоступна: {e}")
                print("[Whisper] Переключаемся на CPU…")
                return _try_load("cpu", compute_type)
        return _try_load(device, compute_type)

    # ── Публичный интерфейс ────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._prev_text = ""
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * BLOCK_DURATION_SEC),
            callback=self._audio_callback,
        )
        self._stream.start()
        print("[Whisper] Запись началась. Говорите…")

    def stop(self) -> None:
        self._running = False
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()
        if self._thread:
            self._thread.join(timeout=5)
        print("[Whisper] Запись остановлена.")

    # ── Аудио-колбэк ──────────────────────────

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[sounddevice] {status}")
        self._audio_queue.put(indata[:, 0].copy())

    # ── Основной цикл ─────────────────────────

    def _process_loop(self) -> None:
        phrase_buffer: list[np.ndarray] = []
        silence_counter = 0.0
        phrase_duration = 0.0

        while self._running:
            try:
                block = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(block ** 2)))
            phrase_buffer.append(block)
            phrase_duration += BLOCK_DURATION_SEC

            if rms < self.silence_threshold:
                silence_counter += BLOCK_DURATION_SEC
            else:
                silence_counter = 0.0

            should_flush = (
                                   silence_counter >= SILENCE_DURATION
                                   and phrase_duration >= MIN_PHRASE_DURATION
                           ) or phrase_duration >= MAX_PHRASE_DURATION

            if should_flush:
                audio = np.concatenate(phrase_buffer)
                phrase_buffer.clear()
                silence_counter = 0.0
                phrase_duration = 0.0
                self._transcribe(audio)

        if phrase_buffer:
            self._transcribe(np.concatenate(phrase_buffer))

    # ── Предобработка аудио ───────────────────

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """Шумоподавление + нормализация громкости."""
        if self.noise_reduce:
            audio = nr.reduce_noise(
                y=audio,
                sr=SAMPLE_RATE,
                stationary=False,  # адаптивное: лучше для фонового шума
                prop_decrease=0.75,  # 75% подавления (100% режет голос)
            )

        # Нормализация пиковой амплитуды → 0.95
        peak = np.abs(audio).max()
        if peak > 1e-6:
            audio = audio / peak * 0.95

        return audio.astype(np.float32)

    # ── Транскрибирование ─────────────────────

    def _transcribe(self, audio: np.ndarray) -> None:
        if len(audio) / SAMPLE_RATE < 0.5:
            return

        audio = self._preprocess(audio)

        # Prompt = тематический подсказ + хвост предыдущих фраз (скользящий контекст)
        prompt_parts = []
        if self.initial_prompt:
            prompt_parts.append(self.initial_prompt)
        if self._prev_text:
            prompt_parts.append(self._prev_text[-CONTEXT_WINDOW_CHARS:])
        prompt = " ".join(prompt_parts) or None

        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self.language,
                initial_prompt=prompt,

                # Параметры декодирования
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE,

                # Фильтрация галлюцинаций
                no_speech_threshold=NO_SPEECH_THRESHOLD,
                log_prob_threshold=LOG_PROB_THRESHOLD,
                compression_ratio_threshold=COMPRESSION_RATIO,
                condition_on_previous_text=True,

                # VAD
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 200,
                },
            )

            # Материализуем генератор внутри try — перехватываем ошибки CTranslate2
            text = " ".join(seg.text.strip() for seg in segments).strip()

        except RuntimeError as e:
            print(f"[Whisper] Ошибка: {e}")
            return
        except Exception as e:
            print(f"[Whisper] {type(e).__name__}: {e}")
            return

        # Фильтр: игнорируем слишком короткие результаты
        if len(text.split()) < MIN_TEXT_TOKENS:
            return

        # Обновляем скользящий контекст
        self._prev_text = (self._prev_text + " " + text).strip()

        self.on_transcript(text)


# ──────────────────────────────────────────────
# CLI / демо
# ──────────────────────────────────────────────

def _demo() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Realtime Whisper transcriber")
    parser.add_argument("--model", default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--lang", default=DEFAULT_LANGUAGE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--compute", default=DEFAULT_COMPUTE_TYPE)
    parser.add_argument(
        "--prompt", default=None,
        help="Тематический подсказ, напр.: 'Слова'"
    )
    parser.add_argument("--no-nr", action="store_true", help="Отключить шумоподавление")
    args = parser.parse_args()

    results: list[str] = []

    def on_text(text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        print(line)
        results.append(line)

    transcriber = RealtimeTranscriber(
        on_transcript=on_text,
        model_size=args.model,
        language=args.lang,
        initial_prompt=args.prompt,
        device=args.device,
        compute_type=args.compute,
        noise_reduce=not args.no_nr,
    )

    transcriber.start()
    try:
        input("\nНажмите Enter для остановки...\n")
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.stop()

    if results:
        print("\n── Итоговый транскрипт ──")
        print("\n".join(results))


if __name__ == "__main__":
    _demo()