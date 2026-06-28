"""Curated metabolite -> pathway -> gene knowledge base.

This is a hand-built, offline approximation of KEGG / SMPDB pathway membership
restricted to the kinds of metabolites seen in HILIC untargeted assays. It is
deliberately conservative: only well-established memberships are encoded so that
the over-representation analysis is interpretable without a network connection.

Three objects are exported:
    PATHWAYS        - pathway -> list of canonical metabolite names
    PATHWAY_GENES   - pathway -> list of human enzyme gene symbols
    METABOLITE_CLASS- canonical metabolite -> coarse chemical class

Name matching is fuzzy (see `normalize_name`) so dataset-specific suffixes such
as "_(principal)", "L-" prefixes, charge/adduct notation, etc. still resolve.
"""
from __future__ import annotations

import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Pathway -> member metabolites (canonical names, lower-case matched later)
# ---------------------------------------------------------------------------
PATHWAYS: dict[str, list[str]] = {
    "Arginine biosynthesis & urea cycle": [
        "urea", "ornithine", "citrulline", "arginine", "argininosuccinic acid",
        "arginosuccinic acid", "n-acetylglutamic acid", "n-acetyl-glutamate",
        "aspartic acid", "n-acetylornithine", "fumaric acid", "homoarginine",
        "n-alpha-acetylarginine", "creatine",
    ],
    "Glycine, serine & threonine metabolism": [
        "glycine", "serine", "l-serine", "threonine", "l-threonine", "sarcosine",
        "dimethylglycine", "betaine", "betaine aldehyde", "choline",
        "guanidinoacetic acid", "guanidoacetic acid", "creatine", "cysteine",
        "o-phosphoethanolamine", "ethanolamine", "n-acetylserine",
    ],
    "Methionine & one-carbon metabolism": [
        "l-methionine", "methionine", "s-adenosyl-l-methionine",
        "s-adenosyl-l-homocysteine", "homocysteine", "homocysteic acid",
        "betaine", "dimethylglycine", "sarcosine", "5-methylthioadenosine",
        "methionine sulfoxide", "cystathionine", "methylcysteine",
    ],
    "Creatine metabolism": [
        "creatine", "creatinine", "guanidinoacetic acid", "guanidoacetic acid",
        "arginine", "glycine", "s-adenosyl-l-methionine",
    ],
    "Choline & TMAO metabolism": [
        "choline", "betaine", "betaine aldehyde", "trimethylamine-n-oxide",
        "glycerophosphocholine", "phosphorylcholine", "acetylcholine",
        "dimethylglycine", "phosphocholine",
    ],
    "Polyamine metabolism": [
        "ornithine", "putrescine", "n-acetylputrescine", "spermidine", "spermine",
        "n-acetyl spermidine", "n-acetyl spermine", "5-methylthioadenosine",
        "agmatine", "s-adenosyl-l-methionine", "arginine",
    ],
    "Nicotinate & NAD metabolism": [
        "nicotinamide", "1-methylnicotinamide", "nicotinamide n-oxide",
        "nicotinamide ribotide", "quinolinic acid", "nicotinic acid",
        "nicotinamide riboside", "acadesine",
    ],
    "Purine metabolism": [
        "adenine", "adenosine", "adenosine monophosphate", "adenosine diphosphate",
        "adenosine triphosphate", "hypoxanthine", "xanthine", "uric acid",
        "inosine", "guanine", "guanosine triphosphate", "allantoin",
        "deoxyadenosine", "deoxyadenosine monophosphate", "1-methyladenine",
        "1-methyladenosine", "adenylsuccinic acid", "deoxyinosine",
        "inosine diphosphate", "inosine triphosphate", "xanthosine",
        "guanine deoxyguanosine diphosphate", "5-methylthioadenosine",
        "aminoimidazole carboxamide ribonucleotide", "purine", "7-methylguanine",
        "7-methylguanosine", "1-methyladenine",
    ],
    "Pyrimidine metabolism": [
        "cytosine", "cytidine", "uracil", "uridine", "thymine", "orotic acid",
        "uridine 5-monophosphate", "cytidine triphosphate", "n-carbamoyl-l-aspartate",
        "orotidine monophosphate", "pseudouridine", "cytidine diphosphate",
        "uridine 5'-monophosphate", "thymidine", "dihydroorotic acid",
    ],
    "TCA cycle": [
        "citric acid", "citric+isocitric acid", "aconitic acid", "a-ketoglutarate",
        "succinic acid", "fumaric acid", "malic acid", "oxaloacetic acid",
        "2-hydroxyglutarate", "itaconic acid", "2-methylcitric acid",
    ],
    "Glycolysis & gluconeogenesis": [
        "pyruvic acid", "l-lactic acid", "3-phosphoglyceric acid", "glyceric acid",
        "glucose-1-phosphate", "acetylphosphate", "glycerol 3-phosphate",
    ],
    "Branched-chain amino acid degradation": [
        "l-valine", "l-leucine + l-isoleucine", "ketoleucine",
        "alpha-ketoisovaleric acid", "2-hydroxy-3-methylbutyric acid",
        "methylmalonic acid", "3-hydroxyisovaleryl carnitine",
        "isovalerylcarnitine", "propionylcarnitine", "2-ketobutyric acid",
        "tiglylcarnitine",
    ],
    "Tryptophan & kynurenine metabolism": [
        "l-tryptophan", "l-kynurenine", "kynurenic acid", "xanthurenic acid",
        "quinolinic acid", "serotonin", "5-hydroxyindoleacetic acid", "melatonin",
        "hydroxykynurenine", "indole", "indoxyl sulfate", "5-methoxytryptophan",
        "anthranilate", "indole-3-carboxylic acid", "indoleacrylic acid",
        "indole-3-acetic acid",
    ],
    "Tyrosine & catecholamine metabolism": [
        "l-tyrosine", "l-phenylalanine", "dopamine", "norepinephrine",
        "normetanephrine", "metanephrine", "homovanillic acid", "tyramine",
        "homogentisic acid", "4-hydroxyphenyllactic acid",
        "p-hydroxyphenylacetic acid", "phenylpyruvate", "phenyllactic acid",
        "phenylalanine",
    ],
    "Histidine metabolism": [
        "l-histidine", "histamine", "1-methylhistidine", "carnosine",
        "imidazoleacetic acid", "histidinol", "urocanic acid",
        "histidinol_(qualifier)", "histidinol_(principal)",
    ],
    "Taurine & sulfur amino-acid metabolism": [
        "taurine", "hypotaurine", "cysteine", "cysteamine", "cystine",
        "methionine sulfoxide", "homocysteic acid", "methylcysteine",
        "pyroglutamic acid",
    ],
    "Glutathione & glutamate metabolism": [
        "l-glutamic acid", "l-glutamine", "pyroglutamic acid", "cysteine",
        "glycine", "5-oxoproline", "2-hydroxyglutarate", "ophthalmic acid",
    ],
    "Proline & collagen-related metabolism": [
        "l-proline", "proline", "hydroxyproline", "1-pyrroline-5-carboxylic acid",
        "l-glutamic acid", "ornithine", "pipecolic acid", "saccharopine",
    ],
    "Carnitine shuttle & fatty-acid beta-oxidation": [
        "l-carnitine", "l-acetylcarnitine", "propionylcarnitine",
        "butyrylcarnitine", "isovalerylcarnitine", "hexanoylcarnitine",
        "octanoylcarnitine", "decanoylcarnitine", "dodecanoylcarnitine",
        "myristoylcarnitine", "l-palmitoylcarnitine", "oleoylcarnitine",
        "linoleylcarnitine", "octadecanoylcarnitine", "malonylcarnitine",
        "tiglylcarnitine", "9-decenoylcarnitine", "dodecenoylcarnitine",
        "3-hydroxyhexanoylcarnitine", "3-hydroxydodecanoylcarnitine",
        "3-hydroxyhexadecanoylcarnitine", "3-hydroxyoleoylcarnitine",
        "3-hydroxylinoleylcarnitine", "l-glutarylcarnitine", "adipoylcarnitine",
        "pimelylcarnitine", "suberylcarnitine", "arachidonyl carnitine",
        "9-hexadecenoylcarnitine", "2-octenoylcarnitine", "hexanoylcarnitine",
    ],
    "Glycerophospholipid metabolism": [
        "choline", "ethanolamine", "o-phosphoethanolamine", "phosphorylcholine",
        "glycerophosphocholine", "glycerol 3-phosphate", "cdp-choline",
    ],
    "Sphingolipid metabolism": [
        "sphingosine 1-phosphate", "sphinganine", "sphingosine", "ceramide",
        "sphingomyelin",
    ],
    "Bile acid biosynthesis": [
        "cholic acid", "deoxycholic acid", "chenodeoxyglycocholic acid",
        "glycocholic acid", "taurocholic acid", "taurodeoxycholic acid",
        "cholesterol", "7-dehydrocholesterol",
    ],
    "Steroid hormone biosynthesis": [
        "cholesterol", "desmosterol", "7-dehydrocholesterol", "zymosterol",
        "testosterone", "cortisol", "corticosterone", "aldosterone",
        "androsterone sulfate", "cholesteryl sulfate", "pregnenolone",
    ],
    "Pantothenate & CoA biosynthesis": [
        "pantothenic acid", "coenzyme a", "coa", "2-ketobutyric acid",
    ],
    "Vitamin B6 metabolism": [
        "pyridoxal", "pyridoxal 5-phosphate", "4-pyridoxic acid", "pyridoxamine",
    ],
    "Pentose phosphate pathway": [
        "sedoheptulose monophosphate", "gluconolactone", "gluconic acid",
        "5-phosphoribosyl-1-pyrophosphate", "ribose 5-phosphate", "6-phosphogluconate",
    ],
    "Beta-alanine & pantothenate metabolism": [
        "beta-alanine", "3-aminoisobutyric acid", "pantothenic acid", "uracil",
        "carnosine",
    ],
}


# ---------------------------------------------------------------------------
# Pathway -> representative human enzyme / transporter gene symbols
# ---------------------------------------------------------------------------
PATHWAY_GENES: dict[str, list[str]] = {
    "Arginine biosynthesis & urea cycle": ["ASS1", "ASL", "ARG1", "ARG2", "OTC", "CPS1", "NOS1", "NOS2", "NOS3"],
    "Glycine, serine & threonine metabolism": ["SHMT1", "SHMT2", "GLDC", "PHGDH", "PSAT1", "PSPH", "GATM", "GAMT", "DMGDH", "SARDH", "BHMT", "CHDH"],
    "Methionine & one-carbon metabolism": ["MAT1A", "MAT2A", "AHCY", "MTR", "MTHFR", "BHMT", "CBS", "CTH", "MTAP"],
    "Creatine metabolism": ["GATM", "GAMT", "CKM", "CKB", "CKMT1A", "CKMT2", "SLC6A8"],
    "Choline & TMAO metabolism": ["CHKA", "CHKB", "PCYT1A", "CHDH", "BHMT", "FMO3", "PEMT", "ACHE", "CHAT"],
    "Polyamine metabolism": ["ODC1", "SRM", "SMS", "AMD1", "SAT1", "PAOX", "SMOX", "AGMAT", "ARG2"],
    "Nicotinate & NAD metabolism": ["NAMPT", "NMNAT1", "NMNAT3", "NNMT", "NADSYN1", "QPRT", "PNP", "AOX1"],
    "Purine metabolism": ["HPRT1", "APRT", "ADA", "PNP", "XDH", "ADSL", "ADSS1", "AMPD1", "NT5E", "IMPDH1", "IMPDH2", "ATIC"],
    "Pyrimidine metabolism": ["CAD", "DHODH", "UMPS", "TYMS", "DPYD", "UPP1", "CDA", "TK1", "RRM1"],
    "TCA cycle": ["CS", "ACO2", "IDH1", "IDH2", "IDH3A", "OGDH", "SUCLA2", "SDHA", "SDHB", "FH", "MDH1", "MDH2", "PC"],
    "Glycolysis & gluconeogenesis": ["HK1", "GAPDH", "PKM", "LDHA", "LDHB", "PGK1", "ENO1", "PCK1", "G6PC", "PGAM1"],
    "Branched-chain amino acid degradation": ["BCAT1", "BCAT2", "BCKDHA", "BCKDHB", "DBT", "ACAD8", "MCCC1", "MUT", "MMUT", "PCCA", "HMGCL"],
    "Tryptophan & kynurenine metabolism": ["TDO2", "IDO1", "KMO", "KYNU", "KYAT1", "HAAO", "QPRT", "TPH1", "DDC", "MAOA", "ASMT"],
    "Tyrosine & catecholamine metabolism": ["TH", "DDC", "DBH", "PNMT", "COMT", "MAOA", "MAOB", "TAT", "HPD", "HGD"],
    "Histidine metabolism": ["HDC", "HAL", "CARNS1", "CNDP1", "HNMT", "AMDHD1"],
    "Taurine & sulfur amino-acid metabolism": ["CDO1", "CSAD", "GAD1", "ADO", "BAAT", "CTH", "CBS"],
    "Glutathione & glutamate metabolism": ["GCLC", "GCLM", "GSS", "GGT1", "GPX1", "GSR", "OPLAH", "GLS", "GLUL"],
    "Proline & collagen-related metabolism": ["PYCR1", "PYCR2", "ALDH18A1", "PRODH", "OAT", "P4HA1", "P4HA2", "PLOD1", "PLOD2"],
    "Carnitine shuttle & fatty-acid beta-oxidation": ["CPT1A", "CPT1B", "CPT2", "SLC25A20", "CRAT", "ACADM", "ACADVL", "ACADL", "HADHA", "HADHB", "ACSL1"],
    "Glycerophospholipid metabolism": ["CHKA", "PCYT1A", "PLA2G2A", "LPCAT1", "PNPLA2", "GPAM", "PLD1", "ETNK1", "PCYT2"],
    "Sphingolipid metabolism": ["SPHK1", "SPHK2", "SGPP1", "CERS2", "CERS5", "SMPD1", "SGMS1", "ASAH1", "UGCG"],
    "Bile acid biosynthesis": ["CYP7A1", "CYP8B1", "CYP27A1", "BAAT", "SLC10A1", "ABCB11", "NR1H4"],
    "Steroid hormone biosynthesis": ["CYP11A1", "CYP17A1", "CYP21A2", "CYP11B1", "CYP11B2", "HSD3B2", "HSD11B1", "STS", "SRD5A1"],
    "Pantothenate & CoA biosynthesis": ["PANK1", "PANK2", "COASY", "PPCS", "PPCDC"],
    "Vitamin B6 metabolism": ["PDXK", "PNPO", "PDXP", "AOX1"],
    "Pentose phosphate pathway": ["G6PD", "PGLS", "PGD", "TKT", "TALDO1", "RPIA", "PRPS1"],
    "Beta-alanine & pantothenate metabolism": ["GADL1", "CARNS1", "CNDP1", "ABAT", "DPYS", "UPB1"],
}


# Coarse chemical class for each pathway (used for class-level summaries).
CLASS_OF_PATHWAY: dict[str, str] = {
    "Arginine biosynthesis & urea cycle": "Amino acids",
    "Glycine, serine & threonine metabolism": "Amino acids",
    "Methionine & one-carbon metabolism": "Amino acids",
    "Creatine metabolism": "Amino acid derivatives",
    "Choline & TMAO metabolism": "Methylamines",
    "Polyamine metabolism": "Polyamines",
    "Nicotinate & NAD metabolism": "Cofactors/vitamins",
    "Purine metabolism": "Nucleotides",
    "Pyrimidine metabolism": "Nucleotides",
    "TCA cycle": "Organic acids",
    "Glycolysis & gluconeogenesis": "Organic acids",
    "Branched-chain amino acid degradation": "Amino acids",
    "Tryptophan & kynurenine metabolism": "Amino acids",
    "Tyrosine & catecholamine metabolism": "Amino acids",
    "Histidine metabolism": "Amino acids",
    "Taurine & sulfur amino-acid metabolism": "Amino acids",
    "Glutathione & glutamate metabolism": "Amino acids",
    "Proline & collagen-related metabolism": "Amino acids",
    "Carnitine shuttle & fatty-acid beta-oxidation": "Acylcarnitines/lipids",
    "Glycerophospholipid metabolism": "Lipids",
    "Sphingolipid metabolism": "Lipids",
    "Bile acid biosynthesis": "Lipids",
    "Steroid hormone biosynthesis": "Lipids",
    "Pantothenate & CoA biosynthesis": "Cofactors/vitamins",
    "Vitamin B6 metabolism": "Cofactors/vitamins",
    "Pentose phosphate pathway": "Carbohydrates",
    "Beta-alanine & pantothenate metabolism": "Amino acid derivatives",
}


# ---------------------------------------------------------------------------
# Lipid-class detection (structured shorthand never appears in PATHWAYS above)
# ---------------------------------------------------------------------------
_LIPID_PREFIXES = {
    "Sphingolipid metabolism": ("sm(", "sm ", "ceramide", "sphingo"),
    "Glycerophospholipid metabolism": ("pc(", "pe(", "ps(", "pg(", "pi(", "pa(",
                                       "lysopc", "plasmalogen", "bmp(", "cl("),
}


def normalize_name(name: str) -> str:
    """Normalize a metabolite name for fuzzy matching against the KB."""
    s = str(name).lower().strip()
    s = re.sub(r"_\((principal|qualifier)\)", "", s)
    s = re.sub(r"\s*\((principal|qualifier)\)", "", s)
    s = s.replace("#", " ")
    s = re.sub(r"-d\d+\b", "", s)               # internal-standard isotope tags
    s = re.sub(r"\s+pool\b", "", s)
    s = s.replace("dl-", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _strip_stereo(s: str) -> str:
    # also try removing a leading L-/D- for amino acids
    return re.sub(r"^[ld]-", "", s)


def _build_lookup() -> dict[str, set[str]]:
    lut: dict[str, set[str]] = defaultdict(set)
    for pw, members in PATHWAYS.items():
        for m in members:
            key = normalize_name(m)
            lut[key].add(pw)
            lut[_strip_stereo(key)].add(pw)
    return lut


_LOOKUP = _build_lookup()


def map_metabolite(name: str) -> list[str]:
    """Return the list of pathways a metabolite belongs to (possibly empty)."""
    norm = normalize_name(name)
    hits: set[str] = set()

    # Structured lipid shorthand first.
    for pw, prefixes in _LIPID_PREFIXES.items():
        if any(norm.startswith(p) or p in norm for p in prefixes):
            hits.add(pw)
    if hits:
        return sorted(hits)

    # Exact / stereo-insensitive lookup.
    if norm in _LOOKUP:
        hits |= _LOOKUP[norm]
    stereo = _strip_stereo(norm)
    if stereo in _LOOKUP:
        hits |= _LOOKUP[stereo]
    if hits:
        return sorted(hits)

    # Substring containment as a last resort (e.g. "carnitine" families).
    for key, pws in _LOOKUP.items():
        if len(key) >= 5 and (key in norm or norm in key):
            hits |= pws
    return sorted(hits)


def all_pathways() -> list[str]:
    return list(PATHWAYS.keys())
