package com.wararaki.sample.stockserviceclient.demo_client

import DemoClientApplication
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.springframework.web.reactive.function.client.WebClient

class DemoClientApplicationTests {

	private val webClient: WebClient = WebClient.builder().build()

	@Test
	fun shouldRetrieveStockPriceFromTheService() {
		var demoClientApplication = DemoClientApplication(webClient)
		val prices = demoClientApplication.pricesFor("SYMBOL")

		Assertions.assertNotNull(prices)
		val fivePrices = prices.take(5)
		Assertions.assertEquals(5, fivePrices.count().block())
		Assertions.assertEquals("SYMBOL", fivePrices.blockFirst()?.symbol)
	}

}
