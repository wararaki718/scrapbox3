package com.example.spbmysql

import org.springframework.data.annotation.Id
import org.springframework.data.relational.core.mapping.Table

@Table("todo")
data class Todo(@Id val id: Long, val description: String, val details: String, val done: Boolean)
