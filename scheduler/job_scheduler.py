import time

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.logger import get_logger

logger = get_logger(__name__)

eastern = pytz.timezone("America/New_York")


def run_theater_run() -> None:
    from agent.theater_agent import TheaterAgent
    logger.info("Scheduled Theater Run triggered")
    try:
        TheaterAgent().run()
    except Exception as e:
        logger.error(f"Scheduled Theater Run failed: {e}", exc_info=True)


def run_ticketing_run() -> None:
    from ticketing.ticketing_agent import TheaterTicketingAgent
    logger.info("Scheduled Theater Ticketing Run triggered")
    try:
        TheaterTicketingAgent().run()
    except Exception as e:
        logger.error(f"Scheduled Theater Ticketing Run failed: {e}", exc_info=True)


def start_scheduler() -> None:
    """Start the APScheduler and block until Ctrl+C.

    Daily schedule (all Eastern):
      10:00 AM — Web Search Run
      10:15 AM — Ticketing Run (Ticketmaster, SeatGeek, Eventbrite, StubHub)
    """
    scheduler = BackgroundScheduler(timezone=eastern)

    scheduler.add_job(
        run_theater_run,
        trigger=CronTrigger(hour=10, minute=0, timezone=eastern),
        id="daily_theater_run",
        name="Daily NYC Theater Web Run",
        replace_existing=True,
    )
    scheduler.add_job(
        run_ticketing_run,
        trigger=CronTrigger(hour=10, minute=15, timezone=eastern),
        id="daily_theater_ticketing_run",
        name="Daily Theater Ticketing Run",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Theater Scheduler started:")
    logger.info("  10:00 AM ET — Web Search Run")
    logger.info("  10:15 AM ET — Ticketing Run")
    logger.info("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down theater scheduler…")
        scheduler.shutdown()
        logger.info("Theater Scheduler stopped")
