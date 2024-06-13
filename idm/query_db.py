import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect(
    "/Users/davidepaglieri/Desktop/repos/nle/nld-nao/nld-nao-unzipped/ttyrecs_nao.db"
)

# Create a cursor object
cursor = conn.cursor()

# Execute a query to get the list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Iterate over the tables and print their column names
for table in tables:
    table_name = table[0]
    print(f"Table: {table_name}")

    # Execute a query to get the column names for the current table
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    # Extract and print column names
    column_names = [column[1] for column in columns]
    print("Column names:", column_names)
    print()  # Print a newline for better readability


# Execute a query to get the number of games per game version
query = """
SELECT version, COUNT(*) as game_count
FROM games
GROUP BY version;
"""

query = """
SELECT name, gameid
FROM games
WHERE version != '3.4.3' AND death = 'ascended' AND role='Val' AND race='Dwa' and align='Law';
"""

query = """
SELECT name, gameid
FROM games
WHERE version != '3.4.3' AND role='Val' AND race='Dwa' AND align='Law';
"""

# Query to check how long a game took to complete
query = """
SELECT turns
FROM games
WHERE gameid = 145461;
"""

# Query to find the winning game withing a certain number of turns
query = """SELECT name, gameid
FROM games
WHERE version != '3.4.3' AND death = 'ascended' AND turns <= 35000 AND role='Val' AND race='Dwa' AND align='Law';
"""

cursor.execute(query)
games_not_in_version_343 = cursor.fetchall()

# Print the results
for game in games_not_in_version_343:
    print(game)

conn.close()
