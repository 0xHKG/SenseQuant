import time
from loguru import logger
from src.config import settings
from src.engine import Engine, is_market_open

def main():
    engine = Engine(settings.symbols)
    engine.start()
    # simple loop
    while True:
        if is_market_open():
            for sym in settings.symbols:
                engine.tick_intraday(sym)
        else:
            logger.info("Market closed; sleeping longer")
            time.sleep(60)
        time.sleep(5)  # throttle
if __name__ == "__main__":
    main()

