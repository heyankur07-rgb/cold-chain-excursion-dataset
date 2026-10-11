# Cold-Chain Pharmaceutical Shipment Excursions (synthetic dataset)

Original synthetic dataset created by the repository owner for an ML benchmark.
`generate_data.py` generates `data.csv` (64,052 rows, 28 columns): pharmaceutical cold-chain
shipments with the binary label `excursion` (the payload left its required temperature range in transit).

- No third-party, scraped, or LLM-generated data is used. Pure numpy and pandas.
- Reproducible: `python generate_data.py --seed <SEED> --out data.csv`
- The seed for the published dataset is withheld so benchmark participants cannot regenerate the labels. Reviewers can request it.

## License

The script and the dataset it generates are released under **CC0 1.0 Universal (Public Domain Dedication)**.
See `LICENSE`. Commercial use and redistribution are permitted.
