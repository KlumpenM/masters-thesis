- Read and understand the source code on some Privacy Machanisms and Aggregators of `pfl` module find out how noise is applied and how noisy updates are handled.
    - Is it applied in a distributed way, where each client adds $N(0,\sigma^2 / N)$ DP noise locally on $N$ user devices?
    - Is there a trusted third-party that adds $N(0, \sigma^2)$ DP noise after aggregation of the updates of the clients?
    - Try to compare loss and accuracy with and without DP to determine how aggregation of noisy statistics is handled