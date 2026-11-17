def estimate_ee_adc2_memory(myadc):
    print()
    print("Entering memory estimator ...", flush=True)
    nao = myadc.mo_coeff.shape[0]
    nmo = myadc.mo_coeff.shape[1]
    o = myadc._nocc
    v = nmo - o

    # PySCF's internal heuristic
    nao_pair = nao * (nao + 1) // 2
    nmo_pair = nmo * (nmo + 1) // 2

    heuristic = (
        max(nao_pair**2, nmo**4) + nmo_pair**2
    ) * 8

    # Persistent MO integral blocks
    blocks = {
        "oooo": o**4,
        "ovoo": o**3 * v,
        "oovv": o**2 * v**2,
        "ovvo": o**2 * v**2,
        "ovvv": o * v**2 * (v + 1) // 2,
    }

    if not myadc.approx_trans_moments:
        blocks["vvvv"] = v**4

    eri_bytes = 8 * sum(blocks.values())

    print(f"Occupied orbitals: {o}")
    print(f"Virtual orbitals:  {v}")
    print(f"AO basis size:     {nao}")
    print()
    print(f"MO integral storage: {eri_bytes / 2**30:.2f} GiB")
    print(f"PySCF heuristic:     {heuristic / 2**30:.2f} GiB")
