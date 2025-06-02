import singlestoredb as s2db

host = "localhost"
user = "root"
password = "secret"

try:
    # On teste sans base spécifique d'abord
    conn = s2db.connect(
        host=host,
        user=user,
        password=password,
    )
    print("✅ Connection to SingleStore server successful.")

    # Vérifie si la base 'campagne_ombre' existe déjà
    with conn.cursor() as cursor:
        cursor.execute("SHOW DATABASES LIKE 'campagne_ombre'")
        exists = cursor.fetchone()
        if exists:
            print("✅ Database 'campagne_ombre' exists.")
        else:
            print("❌ Database 'campagne_ombre' does not exist yet.")
    conn.close()

except Exception as e:
    print("❌ Failed to connect to SingleStore:", e)
