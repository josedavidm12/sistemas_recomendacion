---Crear tabla con peliculas que han sido calificadas por más de XX cantidad de usuarios
drop table if exists ratings_sel;

create table ratings_sel as select movieId,
                         count(*) as cnt_rat
                         from ratings2
                         group by movieId
                         having cnt_rat >10
                         order by cnt_rat desc ;

---Crear tabla sin las peliculas que no tienen generos establecidos
drop table if exists movies_sel;

create table movies_sel as select *
                         from movies2
                         where genres != '(no genres listed)';


---Creación de tablas filtradas
drop table if exists ratings_final;

create table ratings_final as 
select a.userId as user_id,
a.movieId as movie_id,
a.rating as rating,
a.year_ratings as year_ratings,
a.month as month,
a.day as day
from ratings2 a
inner join ratings_sel b
on a.movieId =b.movieId;

drop table if exists movies_final;

create table movies_final as
select a.movieId as movie_id,
a.title as title,
a.genres as genres,
a.year_movies as year_movies
from movies2 a
inner join movies_sel b
on a.movieId =b.movieId;

---Unir tablas de peliculas y calificaciones

drop table if exists df_final;

create table df_final as select
a.*,
b.title,
b.genres,
b.year_movies
from ratings_final a inner join 
movies_final b on a.movie_id = b.movie_id
