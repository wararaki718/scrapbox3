package com.example.spbmysql

import org.springframework.stereotype.Service

@Service
class TodoService(val db: ToDoRepository) {
    fun findTodos(): List<Todo> = db.findTodos()

    fun createTodo(todo: Todo) {
        db.save(todo)
    }
}
