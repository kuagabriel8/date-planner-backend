"""Direct Postgres access to the Supabase-hosted app tables (user_profiles, date_ideas).

This is separate from the LangGraph checkpointer's Postgres (POSTGRES_URI) — that one
holds thread-scoped conversation state, this one holds durable, frontend-visible user data.
"""

import os

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=os.environ["SUPABASE_DB_URI"],
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


def fetch_profile(user_id: str, date_category: str) -> dict | None:
    with _get_pool().connection() as conn:
        return conn.execute(
            """
            select name, address, partner_name, budget, interests, dietary_restrictions
            from user_profiles
            where user_id = %s and date_category = %s
            """,
            (user_id, date_category),
        ).fetchone()


def upsert_profile(user_id: str, date_category: str, profile: dict) -> None:
    with _get_pool().connection() as conn:
        conn.execute(
            """
            insert into user_profiles
                (user_id, date_category, name, address, partner_name, budget, interests, dietary_restrictions, updated_at)
            values
                (%(user_id)s, %(date_category)s, %(name)s, %(address)s, %(partner_name)s, %(budget)s, %(interests)s, %(dietary_restrictions)s, now())
            on conflict (user_id, date_category) do update set
                name = excluded.name,
                address = excluded.address,
                partner_name = excluded.partner_name,
                budget = excluded.budget,
                interests = excluded.interests,
                dietary_restrictions = excluded.dietary_restrictions,
                updated_at = now()
            """,
            {
                "user_id": user_id,
                "date_category": date_category,
                "name": profile.get("name"),
                "address": profile.get("address"),
                "partner_name": profile.get("partner_name"),
                "budget": profile.get("budget"),
                "interests": profile.get("interests") or [],
                "dietary_restrictions": profile.get("dietary_restrictions") or [],
            },
        )


def fetch_date_ideas(user_id: str, date_category: str) -> list[dict]:
    with _get_pool().connection() as conn:
        return conn.execute(
            """
            select id, idea, location, estimated_cost, date_time, notes, status, rating, review, completed_at
            from date_ideas
            where user_id = %s and date_category = %s
            order by created_at
            """,
            (user_id, date_category),
        ).fetchall()


def upsert_date_idea(row_id: str, user_id: str, date_category: str, idea: dict) -> None:
    with _get_pool().connection() as conn:
        conn.execute(
            """
            insert into date_ideas
                (id, user_id, date_category, idea, location, estimated_cost, date_time, notes, status, updated_at)
            values
                (%(id)s, %(user_id)s, %(date_category)s, %(idea)s, %(location)s, %(estimated_cost)s, %(date_time)s, %(notes)s, %(status)s, now())
            on conflict (id) do update set
                idea = excluded.idea,
                location = excluded.location,
                estimated_cost = excluded.estimated_cost,
                date_time = excluded.date_time,
                notes = excluded.notes,
                status = excluded.status,
                updated_at = now()
            """,
            {
                "id": row_id,
                "user_id": user_id,
                "date_category": date_category,
                "idea": idea.get("idea"),
                "location": idea.get("location"),
                "estimated_cost": idea.get("estimated_cost"),
                "date_time": idea.get("date_time"),
                "notes": idea.get("notes") or [],
                "status": idea.get("status") or "idea",
            },
        )
