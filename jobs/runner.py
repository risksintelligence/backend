import logging
import random
import time

from app.services.impact import update_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

COMPONENT_KEYS = ["policy", "analyses", "labs", "media", "community"]


def main() -> None:
    logging.info("RRIO worker started: updating RAS snapshot every 5 minutes")
    while True:
        metric_updates = {
            key: round(random.uniform(0.05, 0.3), 3)
            for key in COMPONENT_KEYS
        }
        snapshot = update_snapshot(metric_updates)
        logging.info("Updated RAS snapshot: composite=%s", snapshot.composite)
        time.sleep(300)


if __name__ == "__main__":
    main()
