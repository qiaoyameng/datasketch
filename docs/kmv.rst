.. _kmv:

KMV
===

A K minimum values (KMV) sketch retains the smallest ``k`` distinct
fixed-width hash values. Before capacity is reached, ``count`` returns the
exact number of retained distinct hashes. Once saturated, it estimates
cardinality using the k-th order statistic::

n ≈ (k - 1) * 2**width / X_k

The approximate relative standard error is ``1 / sqrt(k - 2)``. Larger ``k``
reduces estimation error. A finite-width hash can still collide: distinct
inputs with the same hash are represented once, so the estimate describes
distinct hash values. Use a 64-bit hash for very large inputs.

For similarity, merge the two retained sets down to their smallest common
capacity and compute the fraction of that bottom-k union present in both
sketches. The ratio is a probabilistic estimate once either sketch is
saturated; it is exact when both sketches contain fewer than their capacity
hashes.

Merging uses the smallest common capacity. Sketches with different hash
schemes, seeds, or custom hash functions cannot be merged or compared.

.. code:: python

    from datasketch import KMV

    values1 = [b"a", b"b", b"c", b"d"]
    values2 = [b"c", b"d", b"e", b"f"]

    kmv1 = KMV(k=128, seed=1)
    kmv1.update_batch(values1)
    kmv2 = KMV(k=128, seed=1)
    kmv2.update_batch(values2)

    print(kmv1.count())
    print(kmv1.jaccard(kmv2))

    kmv1.merge(kmv2)
    print(kmv1.count())
