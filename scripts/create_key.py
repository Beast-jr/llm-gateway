"""Create an API user and print the key ONCE.

Run from the project root:  python -m scripts.create_key --name "test" --budget 500
"""
import argparse

from services.auth import create_user


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an API key for the gateway.")
    parser.add_argument("--name", required=True, help="who this key is for, e.g. 'test'")
    parser.add_argument("--budget", type=int, default=10000, help="daily token budget (default 10000)")
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget cannot be negative")

    user, key = create_user(args.name, args.budget)
    print(f"Created user #{user.id} {user.name!r} (daily budget {user.daily_token_budget} tokens)\n")
    print(f"  API key:  {key}\n")
    print("Copy it NOW. It is not stored anywhere and will NOT be shown again.")
    print("If you lose it, create a new key.")


if __name__ == "__main__":
    main()
