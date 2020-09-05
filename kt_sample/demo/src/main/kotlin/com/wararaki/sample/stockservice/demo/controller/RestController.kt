package com.wararaki.sample.stockservice.demo.controller

import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.RestController
import reactor.core.publisher.Flux
import java.time.Duration
import java.time.LocalDateTime
import java.util.concurrent.ThreadLocalRandom

@RestController
class RestController () {
    @GetMapping(value = ["stock/{symbol}"], produces = [MediaType.TEXT_EVENT_STREAM_VALUE])
    fun prices(@PathVariable symbol: String): Flux<StockPrice> {
        return Flux.interval(Duration.ofSeconds(1)).map{ StockPrice(symbol, this.randomStockPrice(), LocalDateTime.now()) }
    }

    private fun randomStockPrice(): Double {
        return ThreadLocalRandom.current().nextDouble(100.0)
    }
}