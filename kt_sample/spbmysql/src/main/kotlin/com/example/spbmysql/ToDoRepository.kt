package com.example.spbmysql

import org.springframework.data.jdbc.repository.query.Query
import org.springframework.data.repository.CrudRepository

interface ToDoRepository : CrudRepository<Todo, Long> {
    @Query("select * from todo")
    fun findTodos(): List<Todo>
}
