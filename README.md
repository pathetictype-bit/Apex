# PNSN — Persistent Neural State Networ
**An experimental AI architecture exploring persistent neural parameter collections and indexed retrieval.**
PNSN (Persistent Neural State Network) is a research project built around a hypothesis: neural knowledge can be organized into independent collections of parameters, allowing learned collections to be preserved while retrieval mechanisms identify which collections are relevant to a given query.
The project is an ongoing experiment. Its purpose is to implement, test, challenge, and evolve this architectural approach rather than claim that every hypothesis has already been proven.
## Core Idea
PNSN explores organizing learned knowledge into independent neural parameter collections.
The architecture incorporates components for collection management, indexing, retrieval, and trainable collection ranking. The intended workflow is to use a query to identify relevant collections and retrieve information from the selected collection or collections.
The central goals include:
- Preserving trained collections without unintentionally modifying their parameters during the training of other collections.
- Using indexing to help locate relevant collections.
- Using the IRM (the project's trainable retrieval-ranking mechanism) to rank collection candidates.
- Investigating whether this approach can support more flexible model organization and reduce interference between independently trained collections.
These are architectural goals and research hypotheses, not claims that every objective has been conclusively demonstrated.
## Current Experimental Results
### IRM: Trained Ranking vs. Identity Baseline
The saved trained IRM was evaluated against an identity-transformation baseline using seven held-out query phrasings that were not exact copies of the training queries. All seven queries encoded without unknown tokens.
For identity-related queries, either `collection_1` or `collection_2` was considered correct.
**Overall top-1 accuracy:**
- **Identity baseline:** 3 out of 7 queries correct (42.9%).
- **Trained IRM:** 4 out of 7 queries correct (57.1%).
**Accuracy on identity-related queries only:**
- **Identity baseline:** 1 out of 5 queries correct (20%).
- **Trained IRM:** 3 out of 5 queries correct (60%).
The trained IRM achieved higher top-1 accuracy than the identity baseline on this small evaluation set. This provides preliminary evidence that it can generalize to some unseen query phrasings.
### Limitations
These results must be interpreted cautiously:
- The evaluation contains only seven manually selected queries.
- The current experiment involves three collections and only five training examples across them: two for `collection_1`, two for `collection_2`, and one for `collection_3`.
- Some held-out queries were still ranked incorrectly.
- The evaluation does not establish robust generalization across larger datasets, different domains, or substantially more diverse queries.
- Further testing is needed to establish whether improvements persist across larger, more representative evaluation sets.
**Conclusion:** IRM has shown promising preliminary behavior, but large-scale generalization has not yet been established.
## An Open Question: Different Collection Architectures
An additional hypothesis to investigate is whether different collections can use different neural network architectures or parameter shapes while participating in the same PNSN system.
This raises several questions:
- Can collections with different network shapes be trained and preserved independently?
- Can the retrieval system rank and select heterogeneous collections correctly?
- Can their outputs be consumed by the downstream inference pipeline, or are compatibility constraints required?
- If compatibility is necessary, can an interface or transformation mechanism support heterogeneous collections?
This experiment remains an open research direction. Compatibility between differently shaped collections has not yet been established.
## Development and AI-Assisted Implementation
PNSN is my architectural research project. I developed the underlying concept, designed the architecture, directed its implementation, and worked through the decisions required to turn the hypothesis into a functioning system.
However, **the code was not written entirely by me**.
When I began implementing PNSN, I used AI assistants to generate substantial portions of the code while I focused on architectural design, reasoning, and implementation direction.
I want to be transparent about that distinction. The architecture and research direction are mine; the implementation was developed with AI assistance.
Consequently, the code may contain bugs, implementation errors, inefficiencies, or discrepancies between the intended architecture and its actual behavior.
Developers are welcome to:
- Inspect and challenge the implementation.
- Report bugs or inconsistencies.
- Propose alternative implementations.
- Submit fixes and improvements.
- Design experiments that test or challenge the architectural hypotheses.
The goal is not to present the implementation as flawless. It is to make the architecture available for scrutiny and further development.
## Contributing
Contributions that improve correctness, reproducibility, performance, documentation, or experimental validation are welcome.
When proposing changes, please distinguish between the original PNSN architecture and your own modifications. Contributors should receive appropriate credit for their work without misrepresenting the origin of the original architecture.
Please include reproducible evidence when reporting bugs or making claims about experimental improvements.
## License
PNSN is distributed under the Apache License 2.0. See the [`LICENSE`](LICENSE) file for the full license text.
Please review the license for its terms governing use, modification, redistribution, and attribution.
## Research Status
PNSN is an evolving experimental architecture, not a finished or comprehensively validated solution.
Its current implementation demonstrates a functioning collection and retrieval workflow, with preliminary evidence of improvement from trained IRM parameters on a small held-out evaluation set. Broader generalization, scalability, and compatibility between heterogeneous collections remain open questions.
The objective is to make the architecture available for independent inspection, reproducible experimentation, criticism, and continued development.
**The architecture is a hypothesis being tested through implementation and experimentation. The results, including failures and limitations, are part of the research.**
