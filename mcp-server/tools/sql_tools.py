"""
SQL Tools - Postgrespro Demo Database (airline flights data)
Source: https://postgrespro.com/community/demodb
Schema: bookings (bookings, tickets, ticket_flights, flights, airports,
        seats, aircrafts, boarding_passes)
"""

import os
from typing import Dict

import psycopg2


def _get_db_connection():
    """Get PostgreSQL connection from DATABASE_URL env var."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return psycopg2.connect(db_url)


def register_tools(mcp):
    """Register all SQL tools with the MCP server."""

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
    def list_schemas(include_system: bool = False, name_pattern: str = None) -> Dict:
        """
        List all schemas in the database.

        Args:
            include_system: Include pg_* and information_schema (default: False)
            name_pattern: Filter schemas by ILIKE pattern (e.g. 'book%')

        Returns:
            result: List of schemas with name, owner, and has_usage
        """
        try:
            conditions = []
            params = []
            if not include_system:
                conditions.append("n.nspname NOT LIKE %s AND n.nspname != %s")
                params.extend(["pg_%", "information_schema"])
            if name_pattern:
                conditions.append("n.nspname ILIKE %s")
                params.append(name_pattern)

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            sql = f"""
                SELECT n.nspname AS schema_name,
                       pg_get_userbyid(n.nspowner) AS owner,
                       has_schema_privilege(n.nspname, 'USAGE') AS has_usage
                FROM pg_namespace n
                {where}
                ORDER BY n.nspname
            """
            with _get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    cols = [desc[0] for desc in cur.description]
                    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            return {"result": rows}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def find_relationships(table_name: str, schema: str = "bookings") -> Dict:
        """
        Find explicit foreign keys and implied relationships for a table.

        Args:
            table_name: Table to analyze
            schema: Schema name (default: 'bookings')

        Returns:
            explicit: List of explicit FK constraints
            implied: List of implied relationships based on column naming (_id / _fk suffix)
        """
        try:
            explicit_sql = """
                SELECT kcu.column_name,
                       ccu.table_name AS foreign_table,
                       ccu.column_name AS foreign_column,
                       'explicit_fk' AS relationship_type
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_schema = %s AND tc.table_name = %s
            """
            implied_sql = """
                WITH source_cols AS (
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                        AND (column_name LIKE '%%_id' OR column_name LIKE '%%_fk')
                )
                SELECT sc.column_name,
                       t.table_name AS foreign_table,
                       'id' AS foreign_column,
                       CASE
                           WHEN sc.column_name = t.table_name || '_id' THEN 'strong_implied'
                           ELSE 'possible_implied'
                       END AS relationship_type
                FROM source_cols sc
                CROSS JOIN information_schema.tables t
                JOIN information_schema.columns c
                    ON c.table_schema = t.table_schema
                    AND c.table_name = t.table_name
                    AND c.column_name = 'id'
                WHERE t.table_schema = %s AND t.table_name != %s
                    AND sc.data_type = c.data_type
            """
            with _get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(explicit_sql, (schema, table_name))
                    cols = [desc[0] for desc in cur.description]
                    explicit = [dict(zip(cols, row)) for row in cur.fetchall()]

                    cur.execute(implied_sql, (schema, table_name, schema, table_name))
                    cols = [desc[0] for desc in cur.description]
                    implied = [dict(zip(cols, row)) for row in cur.fetchall()]
            return {"explicit": explicit, "implied": implied}
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

    print("Registered 7 SQL Tools (list_schemas, list_tables, describe_table, get_row_count, get_foreign_keys, find_relationships, execute_query)")
