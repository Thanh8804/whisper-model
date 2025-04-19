from faster_whisper import WhisperModel

model = WhisperModel("small", compute_type="int8", device="cpu")

lang_names = {
    "af": "Tiếng Afrikaans", "am": "Tiếng Amhara", "ar": "Tiếng Ả Rập", "as": "Tiếng Assam",
    "az": "Tiếng Azerbaijan", "be": "Tiếng Belarus", "bg": "Tiếng Bulgaria", "bn": "Tiếng Bengal",
    "bo": "Tiếng Tây Tạng", "bs": "Tiếng Bosnia", "ca": "Tiếng Catalan", "ceb": "Tiếng Cebuano",
    "ckb": "Tiếng Kurd", "cs": "Tiếng Séc", "cy": "Tiếng Wales", "da": "Tiếng Đan Mạch",
    "de": "Tiếng Đức", "el": "Tiếng Hy Lạp", "en": "Tiếng Anh", "es": "Tiếng Tây Ban Nha",
    "et": "Tiếng Estonia", "eu": "Tiếng Basque", "fa": "Tiếng Ba Tư", "fi": "Tiếng Phần Lan",
    "fr": "Tiếng Pháp", "gl": "Tiếng Galicia", "gu": "Tiếng Gujarat", "he": "Tiếng Do Thái",
    "hi": "Tiếng Hindi", "hr": "Tiếng Croatia", "hu": "Tiếng Hungary", "hy": "Tiếng Armenia",
    "id": "Tiếng Indonesia", "is": "Tiếng Iceland", "it": "Tiếng Ý", "ja": "Tiếng Nhật",
    "jv": "Tiếng Java", "ka": "Tiếng Gruzia", "kk": "Tiếng Kazakhstan", "km": "Tiếng Khmer",
    "kn": "Tiếng Kannada", "ko": "Tiếng Hàn", "la": "Tiếng Latin", "lo": "Tiếng Lào",
    "lt": "Tiếng Lithuania", "lv": "Tiếng Latvia", "mg": "Tiếng Madagascar", "mi": "Tiếng Maori",
    "mk": "Tiếng Macedonia", "ml": "Tiếng Malayalam", "mn": "Tiếng Mông Cổ", "mr": "Tiếng Marathi",
    "ms": "Tiếng Mã Lai", "mt": "Tiếng Malta", "my": "Tiếng Miến Điện", "ne": "Tiếng Nepal",
    "nl": "Tiếng Hà Lan", "no": "Tiếng Na Uy", "pa": "Tiếng Punjab", "pl": "Tiếng Ba Lan",
    "ps": "Tiếng Pashto", "pt": "Tiếng Bồ Đào Nha", "ro": "Tiếng Romania", "ru": "Tiếng Nga",
    "sa": "Tiếng Phạn", "sd": "Tiếng Sindhi", "si": "Tiếng Sinhala", "sk": "Tiếng Slovakia",
    "sl": "Tiếng Slovenia", "so": "Tiếng Somali", "sq": "Tiếng Albania", "sr": "Tiếng Serbia",
    "su": "Tiếng Sunda", "sv": "Tiếng Thụy Điển", "sw": "Tiếng Swahili", "ta": "Tiếng Tamil",
    "te": "Tiếng Telugu", "th": "Tiếng Thái", "tr": "Tiếng Thổ Nhĩ Kỳ", "uk": "Tiếng Ukraina",
    "ur": "Tiếng Urdu", "uz": "Tiếng Uzbekistan", "vi": "Tiếng Việt", "zh": "Tiếng Trung"
}

def detect_language(audio_path: str):
    segments, result = model.transcribe(audio_path)
    code = result.language
    return {
        "language_code": code,
        "language_name": lang_names.get(code, f"Mã không xác định: {code}")
    }
