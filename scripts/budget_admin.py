"""Look at, or reset, a user's token counter for today (UTC). Handy for testing budgets.

  python -m scripts.budget_admin show  --name test
  python -m scripts.budget_admin reset --name test      # same effect as midnight UTC passing: counter back to 0

Find the user with --name (the name you gave create_key) or --user-id (printed by create_key).
"""
import argparse
import os
import sys

from dotenv import load_dotenv
from redis import Redis
from sqlalchemy import select

from models.user import User
from services.budget import seconds_until_reset, usage_key
from services.db import get_session

load_dotenv()


def find_user(name: str | None, user_id: int | None) -> User:
    with get_session() as session:
        query = select(User).where(User.id == user_id) if user_id else select(User).where(User.name == name)
        users = session.execute(query).scalars().all()
    if not users:
        sys.exit("No such user. Check --name / --user-id.")
    if len(users) > 1:
        sys.exit(f"{len(users)} users are called {name!r}; use --user-id instead.")
    return users[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show or reset a user's token usage for today (UTC).")
    parser.add_argument("action", choices=["show", "reset"])
    parser.add_argument("--name")
    parser.add_argument("--user-id", type=int)
    args = parser.parse_args()
    if not (args.name or args.user_id):
        parser.error("give --name or --user-id")
    if not os.getenv("REDIS_URL"):
        sys.exit("REDIS_URL is not set (check your .env).")

    user = find_user(args.name, args.user_id)
    redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    key = usage_key(user.id)

    if args.action == "reset":
        redis.delete(key)
        print(f"Reset today's counter for {user.name!r} (user #{user.id}). It is back to 0.")
    used = int(redis.get(key) or 0)
    hours, minutes = divmod(seconds_until_reset() // 60, 60)
    print(f"{user.name!r} (user #{user.id}): {used} of {user.daily_token_budget} tokens used today (UTC).")
    print(f"Counter key: {key}   resets for real in about {hours}h {minutes}m")


if __name__ == "__main__":
    main()
