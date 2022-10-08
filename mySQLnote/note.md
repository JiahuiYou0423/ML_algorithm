

# DDL
- database operation

   use `databasename` 

   create database `databasename charset utf8`

   drop `databasename`
- table operation

  show tables  
  create table `table name` (name type, name type, name type ...)
  drop table `table name`
- type
   int
   float
   varchar
   data
   timestamp
  
# DML: operation to the table
- INSERT INTO `tablename[(col1, col2, col3,...,col)] values(val1, val2, val3) or [(val1,val2,val3), (val1, val2, val3)]`
   ```
  insert into student(id) values(1), (2), (3);
  insert into student(id, name, age) values(4, 'nameA', 3), (5, 'nameB', 30), (6, 'nameC', 20);
  ```
- DELETE FROM `TABLENAME` WHERE `CONDITON`  # condition:= < > <= !=
  ```commandline
  delete from student where id = 1;
  delete from student where id < 4;
  delete from student # delete all the table
  ```
- UPDATE `TABLENAME` SET `column = colname` where `condition`
   ```
  update student set name = 'newName' where id =4; # if find id =4 change the name as 'newName'
  ```
# DQL: select 
- SELECT `col name1, col name2|*` from `table name`;   # select columns
   ```
      select id, name, age from student;
      select * from student; select all columns
    ```
- SELECT * from `table name` where `condition`
   ```
  select * from student where age>10;
  ```
# DQL
- SELECT `colume name`|`group function` FROM `table name` group by `column name`;  #group by the column name #column name should be accordance 
  - SUM(COLNAME)
  - AVG(COLNAME)
  - MIN(COLNAME)
  - MAX(COLNAME)
  - COUNT(COLNAME)
   ```commandline
   select gender, avg(age) from student group by gender; # calculate the average age as group by gender
   ```
# DQL: order the selected content
- select `column name` | `group function` | * from `table name` order by asc | desc; #asc is default
  ```commandline
   select * from student where age > 20 order by age desc; # sort the items which have age >20 as descending order
   ```
- select `column name` | `group function` | * from `table name` LIMIT `n`|`n,m` ; # only get n lines, or skip n lines and get m lines 

