package com.example.spbmysql

import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController

@RestController
class ToDoController(val service: TodoService) {
    @GetMapping
    fun index(): List<Todo> = service.findTodos()

    @PostMapping
    fun post(@RequestBody todo: Todo) {
        service.createTodo(todo)
    }
}