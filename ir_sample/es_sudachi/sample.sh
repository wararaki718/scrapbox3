#!/bin/bash
curl -X GET "localhost:9200/test_sudachi/_analyze?pretty" -H 'Content-Type: application/json' -d'{"analyzer":"sudachi_analyzer", "text" : "関西国際空港"}'
curl -X GET "localhost:9200/test_sudachi/_analyze?pretty" -H 'Content-Type: application/json' -d'{"analyzer":"sudachi_a_analyzer", "text" : "関西国際空港"}'
echo "DONE"
