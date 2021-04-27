#!/bin/bash
curl -XPUT -H 'Content-Type: application/json' localhost:9200/test_sudachi -d @analysis_sudachi.json
echo "DONE"
