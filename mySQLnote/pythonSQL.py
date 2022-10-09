from pymysql import Connection

# access to the SQL database
conn = Connection(
    host = "localhost",
    port = 3306,
    user = "root",
    password="1234"
)
print(conn.get_server_info())
# Using SQL to execute
# get cursor
cursor = conn.cursor()
conn.select_db("world") # select the database by passing the database name
# use cursor to use SQL
#cursor.execute("create table test_quary(id int);") # can omit ;

# get the select result
cursor.execute("select * from student")
result:tuple = cursor.fetchall()
print(result)

# insert the data
# need commit() to confirm changing the database
cursor.execute("insert into student values(8,'ffgg',38)")
conn.commit()
conn.close()

