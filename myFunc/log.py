import sys

USE_EMOJI = True  # Default, ma può essere disattivato

def log(message, emoji=""):
    prefix = f"{emoji} " if USE_EMOJI and emoji else ""
    print(f"{prefix}{message}")

def main(use_emoji):
    global USE_EMOJI
    if not use_emoji:
        USE_EMOJI = False

if __name__ == "__main__":
    main()
