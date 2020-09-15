package com.wararaki.kotlindemo

import org.springframework.data.repository.CrudRepository

interface BookRepositories : CrudRepository<Book, Long>
{
    fun findByTitle(title : String) : Book
}