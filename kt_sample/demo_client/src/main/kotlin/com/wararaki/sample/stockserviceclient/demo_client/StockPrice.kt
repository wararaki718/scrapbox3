package com.wararaki.sample.stockserviceclient.demo_client

import java.time.LocalDateTime

data class StockPrice (val symbol: String? = null, val price: Double? = null, val time: LocalDateTime? = null)
