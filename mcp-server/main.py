"""MCP server entrypoint for local stdio and HTTP transports."""

from __future__ import annotations

import argparse

import psycopg2

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Multi-Dataset Function Calling Tools", json_response=True)


# ============================================================================
# TOOL IMPORTS - Each module registers its own tools
# ============================================================================

# Import tools - they will auto-register via decorators
from tools import bfcl_math_tools, jefferson_stats_tools

# Pass the mcp instance to each module for registration
bfcl_math_tools.register_tools(mcp)
jefferson_stats_tools.register_tools(mcp)


# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="MCP server for function calling threshold testing."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport type for the server.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind for HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Optional ASGI mount path for HTTP transports.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.transport != "stdio":
        mcp.settings.host = args.host
        mcp.settings.port = args.port

    mcp.run(transport=args.transport, mount_path=args.mount_path)


def _get_db_connection():
    """Get PostgreSQL connection from DATABASE_URL env var."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return psycopg2.connect(db_url)


# ============================================================================
# POSTGRESQL TOOLS - Postgrespro Demo Database (airline flights data)
# Source: https://postgrespro.com/community/demodb
# Schema: bookings (bookings, tickets, ticket_flights, flights, airports,
#         seats, aircrafts, boarding_passes)
# ============================================================================

@mcp.tool()
def list_tables(schema: str = "bookings") -> Dict:
    """
    List all tables in a database schema.

    Args:
        schema: Schema name to list tables from (default: 'bookings')

    Returns:
        result: List of table names in the schema
    """
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = %s ORDER BY table_name",
                    (schema,)
                )
                tables = [row[0] for row in cur.fetchall()]
        return {"result": tables}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def describe_table(table_name: str, schema: str = "bookings") -> Dict:
    """
    Get the column names and types for a database table.

    Args:
        table_name: Name of the table to describe
        schema: Schema name (default: 'bookings')

    Returns:
        result: List of column definitions with name and data type
    """
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
                    (schema, table_name)
                )
                cols = [{"column": row[0], "type": row[1]} for row in cur.fetchall()]
        return {"result": cols}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_row_count(table_name: str, schema: str = "bookings") -> Dict:
    """
    Get the total number of rows in a database table.

    Args:
        table_name: Name of the table
        schema: Schema name (default: 'bookings')

    Returns:
        result: Row count as integer
    """
    try:
        from psycopg2 import sql as psql
        with _get_db_connection() as conn:
            with conn.cursor() as cur:
                query = psql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    psql.Identifier(schema),
                    psql.Identifier(table_name)
                )
                cur.execute(query)
                count = cur.fetchone()[0]
        return {"result": count}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_foreign_keys(table_name: str, schema: str = "bookings") -> Dict:
    """
    Get the foreign key relationships for a database table.

    Args:
        table_name: Name of the table
        schema: Schema name (default: 'bookings')

    Returns:
        result: List of foreign key relationships
    """
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT kcu.column_name, ccu.table_name AS foreign_table,
                           ccu.column_name AS foreign_column
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND tc.table_schema = %s AND tc.table_name = %s
                    """,
                    (schema, table_name)
                )
                fks = [
                    {"column": r[0], "references_table": r[1], "references_column": r[2]}
                    for r in cur.fetchall()
                ]
        return {"result": fks}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def execute_query(sql: str, limit: int = 50) -> Dict:
    """
    Execute a read-only SQL SELECT query on the database.

    Args:
        sql: SQL SELECT query to execute
        limit: Maximum number of rows to return (default: 50)

    Returns:
        result: List of rows as dictionaries
    """
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cur:
                safe_sql = "SELECT * FROM (%s) subq LIMIT %s"
                cur.execute(safe_sql, (psycopg2.extensions.AsIs(sql), limit))
                cols = [desc[0] for desc in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return {"result": rows, "row_count": len(rows)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    main()
