package com.wararaki.sample.stockservice.demo.controller

import java.time.LocalDateTime

data class StockPrice (val symbol: String, val price: Double, val time: LocalDateTime)
