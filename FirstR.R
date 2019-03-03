#comment
#Напишите функцию get_fractions, которая принимает на вход два числа, m и n, 
#и возвращает аналогичный вектор, содержащий все дроби вида {i/m, i = 0, 1, ..., m} и  {j/n, j = 0, 1, ..., n}. 
#Вектор не должен содержать повторов. И -- сюжетный поворот -- должен быть упорядочен в порядке убывания.

get_fractions <- function(m, n) {
  i <- seq(0,m)
  j <- seq(0,n)
  m_vect <- i/m
  n_vect <- j/n
  total_vect <- c(m_vect,n_vect)
  #print(total_vect)
  total_vect <- sort(c(m_vect,n_vect),decreasing = TRUE)
  #print(total_vect)
  total_vect <- unique(sort(c(m_vect,n_vect),decreasing = TRUE))
  print(total_vect)
  #m_vect <- seq()
}