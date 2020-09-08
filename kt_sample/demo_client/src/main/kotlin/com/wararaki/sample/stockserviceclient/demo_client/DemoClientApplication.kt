import com.wararaki.sample.stockserviceclient.demo_client.StockPrice
import org.springframework.web.reactive.function.client.WebClient
import reactor.core.publisher.Flux

class DemoClientApplication (_webClient: WebClient){
	private val webClient: WebClient = _webClient

	fun pricesFor(symbol: String): Flux<StockPrice> {
		return webClient.get()
				.uri("http://localhost:8080/stock/{symbol}", symbol)
				.retrieve()
				.bodyToFlux(StockPrice::class.java)
	}
}

