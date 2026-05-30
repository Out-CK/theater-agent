from __future__ import annotations

from typing import List, Literal

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert at generating search queries to find upcoming NYC theater shows.
Your task is to produce exactly 40 search queries that will surface the widest possible range of
upcoming Broadway, off-Broadway, and off-off-Broadway theater productions in New York City.

Rules:
- Generate EXACTLY 40 queries — no more, no fewer.
- Each query must have a query_type of "broad" or "niche".
- Broad queries (~15): General searches like "Broadway shows NYC this week",
  "upcoming Broadway performances 2026", "off-Broadway shows NYC this month",
  "best theater shows NYC", "new plays opening Broadway", "theater events New York City",
  "limited engagement Broadway show", "NYC theater this weekend", "Broadway musicals running now",
  "what's playing on Broadway", "new Broadway show opening", "NYC stage productions upcoming",
  "off-off-Broadway shows NYC", "preview week Broadway", "Tony Award nominees running Broadway".
- Niche queries (~25): Venue-specific or production-specific. Must include dedicated queries
  for each of these venues and organizations:
    BROADWAY THEATERS (~12 queries): Shubert Theatre NYC, Booth Theatre Broadway,
      Music Box Theatre NYC, Imperial Theatre Broadway, St. James Theatre NYC,
      Gershwin Theatre NYC, Richard Rodgers Theatre NYC, Majestic Theatre Broadway,
      Al Hirschfeld Theatre Broadway, Brooks Atkinson Theatre NYC,
      Minskoff Theatre NYC, August Wilson Theatre Broadway
    OFF-BROADWAY VENUES (~8 queries): The Public Theater NYC, Playwrights Horizons NYC,
      Manhattan Theatre Club NYC, New York Theatre Workshop, Second Stage Theatre NYC,
      Atlantic Theater Company NYC, Roundabout Theatre Company NYC, Signature Theatre NYC
    FORMATS (~5 queries): Broadway musical NYC, new play opening Broadway,
      limited engagement Broadway, off-Broadway drama NYC, preview week Broadway show
- Output a JSON array of objects, each with "query" (string) and "query_type" ("broad" or "niche").
"""


class SearchQuery(BaseModel):
    query: str
    query_type: Literal["broad", "niche"]


class SearchPlan(BaseModel):
    queries: List[SearchQuery]


class SearchPlanAgent:
    def __init__(self):
        self._llm = ChatAnthropic(model=MODEL).with_structured_output(SearchPlan)

    def generate(self) -> SearchPlan:
        logger.info("Generating Theater Search Plan…")
        for attempt in range(2):
            try:
                plan: SearchPlan = self._llm.invoke(
                    [{"role": "user", "content": SYSTEM_PROMPT}]
                )
                if len(plan.queries) != 40:
                    logger.warning(
                        f"Search plan returned {len(plan.queries)} queries (expected 40). "
                        f"Attempt {attempt + 1}/2."
                    )
                    if attempt == 1:
                        raise ValueError(
                            f"LLM produced {len(plan.queries)} queries after 2 attempts; expected 40."
                        )
                    continue
                logger.info(f"Theater Search Plan generated with {len(plan.queries)} queries")
                for i, q in enumerate(plan.queries, 1):
                    logger.debug(f"  [{i:02d}] [{q.query_type}] {q.query}")
                return plan
            except ValueError:
                raise
            except Exception as e:
                logger.error(f"Search Plan LLM call failed on attempt {attempt + 1}: {e}")
                if attempt == 1:
                    raise
        raise RuntimeError("Search Plan generation failed unexpectedly")
