"""Uvicorn entry point for the MMP Explorer web UI."""

import argparse
import os
import sys


def main():
    import uvicorn
    from mmp.web import app

    parser = argparse.ArgumentParser(description="MMP Explorer web server")
    parser.add_argument(
        "--db-dir",
        default=".",
        help="Directory to scan for .duckdb files (default: current dir)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    db_dir = os.path.abspath(args.db_dir)
    if not os.path.isdir(db_dir):
        print(f"Error: --db-dir {db_dir!r} is not a directory", file=sys.stderr)
        sys.exit(1)

    app.state.db_dir = db_dir
    print(f"MMP Explorer  db_dir={db_dir}  http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
