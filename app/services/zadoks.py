"""Échelle de Zadoks (orge) — stades 00 à 99 + floraison parallèle, reliés à la ST.

Définitions posées avec l'utilisateur :
- TR (température de référence / zéro de végétation orge) = 0 °C
- TM (température moyenne du jour) = (Tmax + Tmin) / 2
- DJ (degré jour de croissance) = TM - TR (en °C, plancher à 0)
- ST (somme des températures) = somme des DJ depuis le semis (en °C)

Stades appliqués :
- Germination (Zadoks 00-09, ST 0 à <30 °C, puis levée 30 à <100 °C) :
  - 00 : semence sèche (caryopse sec), ST = 0
  - 03 : imbibition complète de la graine (interpolé ~10 °C)
  - 05 : élongation de la radicule, apparition des poils absorbants et
    développement des racines secondaires (interpolé ~18 °C ; 2e « 03 » reçu
    interprété comme 05 selon Zadoks standard)
  - 07 : le coléoptile émerge de la graine (interpolé ~24 °C)
  - 09 : levée — le coléoptile perce la surface du sol, ST = 30 °C
    (ex : 3 °C/j × 10 j, 5 °C/j × 6 j, 10 °C/j × 3 j)
  - 10 : première feuille à travers le coléoptile, pas encore étalée
    (interpolé ~60 °C) — fin de levée, aucune feuille étalée.
- Développement des feuilles (Zadoks 11-19), à partir de ST = 100 °C : première
  feuille sortie du coléoptile, 1ère feuille étalée (Zadoks 11), puis
  1 feuille supplémentaire tous les ~25 °C jusqu'à ST = 300 °C : 9 feuilles
  ou plus étalées (Zadoks 19).
- 20 : aucune talle visible (seuil interpolé à 350 °C, à confirmer — entre
  19 à 300 °C et 21 à 450 °C).
- Tallage : 1ère talle à ST = 450 °C (21), 2 talles vers 500 °C (22, interpolé),
  3 talles à ST = 550 °C (23), puis +1 talle tous les ~25 °C jusqu'à ST = 700 °C
  : nombre maximum de talles visibles (29).
- Montaison : 30 à ST = 800 °C (épi à 1 cm au-dessus du plateau de tallage),
  31 : 1er nœud à 1 cm max au-dessus du plateau, 32 à ST = 950 °C (2e nœud à
  2 cm max au-dessus du 1er), puis +1 nœud tous les ~25 °C (interpolé) jusqu'à
  37 (dernière feuille juste visible, enroulée), 39 (limbe de la dernière
  feuille entièrement étalé, ligule visible).
- Gonflement (stade principal 4, dès 1100 °C) : 41 début (élongation de la gaine
  de la dernière feuille, à 1100 °C), 43 milieu (gaine visiblement gonflée,
  interpolé à 1175 °C), 45 fin (gonflement maximal, interpolé à 1200 °C),
  47 ouverture de la gaine (interpolé à 1225 °C), 49 premières arêtes
  (barbes) visibles à ST = 1250 °C.
- Épiaison (stade principal 5) : 51 à ST = 1400 °C (extrémité de
  l'inflorescence sortie de la gaine), 52 : 20 % sortie (interpolé à 1430 °C),
  53 : 30 % (interpolé à 1445 °C), 54 : 40 % (interpolé à 1460 °C),
  55 : 50 % (interpolé à 1480 °C), 56 : 60 % (interpolé à 1500 °C),
  57 : 70 % (interpolé à 1515 °C), 58 : 80 % (interpolé à 1530 °C),
  59 : fin de l'épiaison à ST = 1550 °C.
- Floraison (stade principal 6, EN PARALLÈLE de l'épiaison chez l'orge) :
  61 à ST = 1450 °C (début : premières anthères sorties), 65 pleine floraison
  (50 % des anthères, interpolé à 1500 °C), 69 fin de floraison (interpolé à
  1550 °C). Le stade 7 commence à ST = 1600 °C.
- Formation du grain (stade principal 7, dès 1600 °C) : 71 stade aqueux
  (premières graines à moitié de leur taille finale, à 1600 °C), 73 début du
  stade laiteux (interpolé à 1660 °C), 75 mi-laiteux (contenu laiteux, graines
  à taille finale mais vertes, interpolé à 1720 °C), 77 fin du stade laiteux
  (interpolé à 1760 °C). Prochain stade à ST = 1800 °C.
- Maturation (stade principal 8, dès 1800 °C) : 83 début du stade pâteux
  (à 1800 °C), 85 pâteux mou — contenu tendre mais sec, empreinte à l'ongle
  réversible (interpolé à 1865 °C), 87 pâteux dur — contenu dur, empreinte
  irréversible (interpolé à 1930 °C), 89 maturation complète — caryopse dur,
  difficile à couper en deux avec les ongles, à ST = 2000 °C.
- Sénescence / Récolte (dès 2000–2100 °C) : 89 maturation complète (à 2000 °C),
  92 sur-maturité (à 2050 °C), 99 produit récolté (à 2100 °C).

La ST doit toujours être comptée DEPUIS LE SEMIS, pas sur une fenêtre calendaire
arbitraire (le cumul GDD 365 j du backend ne convient pas tel quel).
"""

from __future__ import annotations

BARLEY_TR = 0.0

# (seuil ST inclus, code Zadoks, libellé)
EARLY_STAGES: list[tuple[float, str, str]] = [
    (0, "00", "Germination : semence sèche (caryopse sec)"),
    (10, "03", "Germination : imbibition complète de la graine"),
    (18, "05", "Germination : élongation de la radicule, poils absorbants et racines secondaires"),
    (24, "07", "Germination : le coléoptile émerge de la graine"),
    (30, "09", "Germination : levée — le coléoptile perce la surface du sol"),
    (60, "10", "Germination : première feuille à travers le coléoptile (pas encore étalée)"),
    (100, "11", "Développement des feuilles : 1ère feuille étalée"),
    (125, "12", "Développement des feuilles : 2 feuilles étalées"),
    (150, "13", "Développement des feuilles : 3 feuilles étalées"),
    (175, "14", "Développement des feuilles : 4 feuilles étalées"),
    (200, "15", "Développement des feuilles : 5 feuilles étalées"),
    (225, "16", "Développement des feuilles : 6 feuilles étalées"),
    (250, "17", "Développement des feuilles : 7 feuilles étalées"),
    (275, "18", "Développement des feuilles : 8 feuilles étalées"),
    (300, "19", "Développement des feuilles : 9 feuilles ou plus étalées"),
    (350, "20", "Aucune talle visible (début tallage)"),
    (450, "21", "1 talle visible (début tallage)"),
    (500, "22", "2 talles visibles"),
    (550, "23", "3 talles visibles"),
    (575, "24", "4 talles visibles"),
    (600, "25", "5 talles visibles"),
    (625, "26", "6 talles visibles"),
    (650, "27", "7 talles visibles"),
    (675, "28", "8 talles visibles"),
    (700, "29", "Nombre maximum de talles visibles (fin tallage)"),
    (800, "30", "Début montaison : épi à 1 cm au-dessus du plateau de tallage"),
    (850, "31", "1er nœud à 1 cm max au-dessus du plateau de tallage"),
    (950, "32", "2e nœud à 2 cm max au-dessus du 1er"),
    (975, "33", "3e nœud au-dessus du 2e"),
    (1000, "34", "4e nœud au-dessus du 3e"),
    (1025, "35", "5e nœud au-dessus du 4e"),
    (1050, "36", "6e nœud au-dessus du 5e"),
    (1075, "37", "Dernière feuille juste visible, enroulée sur elle-même"),
    (1085, "38", "Dernière feuille dégagée, limbe en cours d'étalement"),
    (1095, "39", "Limbe de la dernière feuille entièrement étalé, ligule visible"),
    (1100, "41", "Début du gonflement : élongation de la gaine de la dernière feuille"),
    (1175, "43", "Milieu du gonflement : gaine de la dernière feuille visiblement gonflée"),
    (1200, "45", "Fin du gonflement : gonflement maximal de la gaine"),
    (1225, "47", "La gaine de la dernière feuille s'ouvre"),
    (1250, "49", "Premières arêtes (barbes) visibles (fin du gonflement)"),
    (1400, "51", "Début de l'épiaison : extrémité de l'inflorescence sortie de la gaine"),
    (1430, "52", "20 % de l'inflorescence sortie"),
    (1445, "53", "30 % de l'inflorescence sortie"),
    (1460, "54", "40 % de l'inflorescence sortie"),
    (1480, "55", "50 % de l'inflorescence sortie (mi-épiaison)"),
    (1500, "56", "60 % de l'inflorescence sortie"),
    (1515, "57", "70 % de l'inflorescence sortie"),
    (1530, "58", "80 % de l'inflorescence sortie"),
    (1550, "59", "Fin de l'épiaison : inflorescence entièrement sortie"),
    (1600, "71", "Stade aqueux : premières graines à moitié de leur taille finale"),
    (1660, "73", "Début du stade laiteux"),
    (1720, "75", "Mi-laiteux : contenu laiteux, graines à taille finale mais vertes"),
    (1760, "77", "Fin du stade laiteux"),
    (1800, "83", "Début du stade pâteux"),
    (1865, "85", "Pâteux mou : contenu tendre mais sec, empreinte à l'ongle réversible"),
    (1930, "87", "Pâteux dur : contenu dur, empreinte à l'ongle irréversible"),
    (2000, "89", "Maturation complète : caryopse dur, difficile à couper en deux"),
    (2050, "92", "Sur-maturité : caryopse très dur, ne peut être marqué à l'ongle"),
    (2100, "99", "Produit récolté"),
]

# Floraison (stade principal 6) : chez l'orge elle se déroule EN PARALLÈLE de
# l'épiaison — elle n'apparaît donc pas dans la séquence principale ci-dessus
# (qui reste l'épiaison), mais flowering_for_st() la renvoie en complément.
FLOWERING_STAGES: list[tuple[float, str, str]] = [
    (1450, "61", "Début de la floraison : premières anthères sorties"),
    (1500, "65", "Pleine floraison : 50 % des anthères sorties"),
    (1550, "69", "Fin de la floraison"),
]

STAGE7_START = 1600.0

# Détail morphologique par stade (évolutions décrites, pas un simple seuil °C).
# st_source = "user" (ST fournie) ou "interpolated" (estimation à confirmer).
STAGE_META: dict[str, dict] = {
    "00": {"st_source": "user", "organ": "semence (caryopse)", "evolution": "semence sèche, aucune activité visible", "phase": "Germination"},
    "03": {"st_source": "interpolated", "organ": "graine", "evolution": "imbibition complète de la graine", "phase": "Germination"},
    "05": {"st_source": "interpolated", "organ": "radicule / racines", "evolution": "élongation de la radicule, apparition des poils absorbants et développement des racines secondaires", "phase": "Germination"},
    "07": {"st_source": "interpolated", "organ": "coléoptile", "evolution": "le coléoptile émerge de la graine (sous la surface)", "phase": "Germination"},
    "09": {"st_source": "user", "organ": "coléoptile", "evolution": "levée : le coléoptile perce la surface du sol (ST = 30 °C)", "phase": "Germination : levée"},
    "10": {"st_source": "interpolated", "organ": "1ère feuille (pointe)", "evolution": "première feuille à travers le coléoptile, pas encore étalée — fin de levée (30 à <100 °C)", "phase": "Germination : levée"},
    "11": {"st_source": "user", "organ": "1ère feuille", "evolution": "développement des feuilles : première feuille sortie du coléoptile et étalée", "phase": "Développement des feuilles"},
    "12": {"st_source": "interpolated", "organ": "2e feuille", "evolution": "développement des feuilles : 2e feuille étalée", "phase": "Développement des feuilles"},
    "13": {"st_source": "interpolated", "organ": "3e feuille", "evolution": "développement des feuilles : 3e feuille étalée", "phase": "Développement des feuilles"},
    "14": {"st_source": "interpolated", "organ": "4e feuille", "evolution": "développement des feuilles : 4e feuille étalée", "phase": "Développement des feuilles"},
    "15": {"st_source": "interpolated", "organ": "5e feuille", "evolution": "développement des feuilles : 5e feuille étalée", "phase": "Développement des feuilles"},
    "16": {"st_source": "interpolated", "organ": "6e feuille", "evolution": "développement des feuilles : 6e feuille étalée", "phase": "Développement des feuilles"},
    "17": {"st_source": "interpolated", "organ": "7e feuille", "evolution": "développement des feuilles : 7e feuille étalée", "phase": "Développement des feuilles"},
    "18": {"st_source": "interpolated", "organ": "8e feuille", "evolution": "développement des feuilles : 8e feuille étalée", "phase": "Développement des feuilles"},
    "19": {"st_source": "user", "organ": "9e feuille et suivantes", "evolution": "développement des feuilles : 9 feuilles ou plus étalées", "phase": "Développement des feuilles"},
    "20": {"st_source": "interpolated", "organ": "talles", "evolution": "aucune talle visible (début tallage)"},
    "21": {"st_source": "user", "organ": "1ère talle", "evolution": "début du tallage : 1ère talle visible"},
    "22": {"st_source": "interpolated", "organ": "talles", "evolution": "2 talles visibles"},
    "23": {"st_source": "user", "organ": "talles", "evolution": "3 talles visibles"},
    "24": {"st_source": "interpolated", "organ": "talles", "evolution": "4 talles visibles"},
    "25": {"st_source": "interpolated", "organ": "talles", "evolution": "5 talles visibles"},
    "26": {"st_source": "interpolated", "organ": "talles", "evolution": "6 talles visibles"},
    "27": {"st_source": "interpolated", "organ": "talles", "evolution": "7 talles visibles"},
    "28": {"st_source": "interpolated", "organ": "talles", "evolution": "8 talles visibles"},
    "29": {"st_source": "user", "organ": "talles", "evolution": "nombre maximum de talles visibles (fin tallage)"},
    "30": {"st_source": "user", "organ": "épi / tige", "evolution": "début montaison : épi à 1 cm au-dessus du plateau de tallage"},
    "31": {"st_source": "interpolated", "organ": "1er nœud", "evolution": "1er nœud à 1 cm max au-dessus du plateau de tallage"},
    "32": {"st_source": "user", "organ": "2e nœud", "evolution": "2e nœud à 2 cm max au-dessus du 1er"},
    "33": {"st_source": "interpolated", "organ": "3e nœud", "evolution": "3e nœud au-dessus du 2e"},
    "34": {"st_source": "interpolated", "organ": "4e nœud", "evolution": "4e nœud au-dessus du 3e"},
    "35": {"st_source": "interpolated", "organ": "5e nœud", "evolution": "5e nœud au-dessus du 4e"},
    "36": {"st_source": "interpolated", "organ": "6e nœud", "evolution": "6e nœud au-dessus du 5e"},
    "37": {"st_source": "user", "organ": "dernière feuille", "evolution": "dernière feuille juste visible, enroulée sur elle-même"},
    "38": {"st_source": "interpolated", "organ": "dernière feuille", "evolution": "dernière feuille dégagée, limbe en cours d'étalement"},
    "39": {"st_source": "user", "organ": "dernière feuille (limbe + ligule)", "evolution": "limbe entièrement étalé, ligule visible"},
    "41": {"st_source": "user", "organ": "gaine de la dernière feuille", "evolution": "début du gonflement par élongation de la gaine"},
    "43": {"st_source": "interpolated", "organ": "gaine de la dernière feuille", "evolution": "milieu du gonflement : gaine visiblement gonflée"},
    "45": {"st_source": "interpolated", "organ": "gaine de la dernière feuille", "evolution": "fin du gonflement : gonflement maximal"},
    "47": {"st_source": "interpolated", "organ": "gaine de la dernière feuille", "evolution": "la gaine s'ouvre"},
    "49": {"st_source": "user", "organ": "arêtes (barbes)", "evolution": "premières arêtes visibles (fin du gonflement)"},
    "51": {"st_source": "user", "organ": "inflorescence (épi)", "evolution": "début de l'épiaison : extrémité sortie de la gaine"},
    "52": {"st_source": "interpolated", "organ": "inflorescence", "evolution": "20 % de l'inflorescence sortie"},
    "53": {"st_source": "interpolated", "organ": "inflorescence", "evolution": "30 % de l'inflorescence sortie"},
    "54": {"st_source": "interpolated", "organ": "inflorescence", "evolution": "40 % de l'inflorescence sortie"},
    "55": {"st_source": "interpolated", "organ": "inflorescence", "evolution": "50 % sortie (mi-épiaison)"},
    "56": {"st_source": "interpolated", "organ": "inflorescence", "evolution": "60 % de l'inflorescence sortie"},
    "57": {"st_source": "interpolated", "organ": "inflorescence", "evolution": "70 % de l'inflorescence sortie"},
    "58": {"st_source": "interpolated", "organ": "inflorescence", "evolution": "80 % de l'inflorescence sortie"},
    "59": {"st_source": "user", "organ": "inflorescence", "evolution": "fin de l'épiaison : inflorescence entièrement sortie"},
    "61": {"st_source": "user", "organ": "anthères", "evolution": "début de la floraison : premières anthères sorties"},
    "65": {"st_source": "interpolated", "organ": "anthères", "evolution": "pleine floraison : 50 % des anthères sorties"},
    "69": {"st_source": "interpolated", "organ": "anthères", "evolution": "fin de la floraison"},
    "71": {"st_source": "user", "organ": "graines", "evolution": "stade aqueux : premières graines à moitié de leur taille finale"},
    "73": {"st_source": "interpolated", "organ": "graines", "evolution": "début du stade laiteux"},
    "75": {"st_source": "interpolated", "organ": "graines", "evolution": "mi-laiteux : contenu laiteux, graines à taille finale mais vertes"},
    "77": {"st_source": "interpolated", "organ": "graines", "evolution": "fin du stade laiteux"},
    "83": {"st_source": "user", "organ": "graines", "evolution": "début du stade pâteux"},
    "85": {"st_source": "interpolated", "organ": "graines", "evolution": "pâteux mou : contenu tendre mais sec, empreinte à l'ongle réversible"},
    "87": {"st_source": "interpolated", "organ": "graines", "evolution": "pâteux dur : contenu dur, empreinte à l'ongle irréversible"},
    "89": {"st_source": "user", "organ": "caryopse", "evolution": "maturation complète : caryopse dur, difficile à couper en deux"},
    "92": {"st_source": "user", "organ": "caryopse", "evolution": "sur-maturité : caryopse très dur, ne peut être marqué à l'ongle"},
    "93": {"st_source": "interpolated", "organ": "graines", "evolution": "des graines se détachent pendant la journée"},
    "97": {"st_source": "interpolated", "organ": "plante entière", "evolution": "la plante meurt et s'affaisse"},
    "99": {"st_source": "interpolated", "organ": "produit", "evolution": "produit récolté"},
}

FLOWERING_META: dict[str, dict] = {
    "61": STAGE_META["61"],
    "65": STAGE_META["65"],
    "69": STAGE_META["69"],
}


def mean_temp(tmax: float, tmin: float) -> float:
    """TM = (Tmax + Tmin) / 2."""
    return (tmax + tmin) / 2.0


def degree_day(tmax: float, tmin: float, tr: float = BARLEY_TR) -> float:
    """DJ = TM - TR, plancher à 0 (pas de croissance sous TR)."""
    return max(0.0, mean_temp(tmax, tmin) - tr)


def thermal_sum(daily: list[dict]) -> float:
    """ST = somme des DJ d'une liste de relevés {tmax, tmin} ou {dj}."""
    total = 0.0
    for day in daily:
        if isinstance(day.get("dj"), (int, float)):
            total += max(0.0, float(day["dj"]))
        elif isinstance(day.get("tmax"), (int, float)) and isinstance(day.get("tmin"), (int, float)):
            total += degree_day(float(day["tmax"]), float(day["tmin"]))
    return round(total * 10) / 10


def stage_for_st(st: float | None) -> dict:
    """Retourne le stade Zadoks (00–99) correspondant à une ST depuis le semis."""
    if st is None or not isinstance(st, (int, float)) or st < 0:
        return {"code": None, "label": None, "principal": None, "sub": None, "st": st}
    code, label = EARLY_STAGES[0][1], EARLY_STAGES[0][2]
    threshold_hit = EARLY_STAGES[0][0]
    for threshold, c, lab in EARLY_STAGES:
        if st >= threshold:
            code, label, threshold_hit = c, lab, threshold
        else:
            break
    result = {
        "code": code,
        "label": label,
        "principal": int(code[0]) if len(code) == 2 and code.isdigit() else None,
        "sub": int(code[1]) if len(code) == 2 and code.isdigit() else None,
        "st": st,
        "stThreshold": threshold_hit,
        "stBeyond": round((st - threshold_hit) * 10) / 10,
    }
    meta = STAGE_META.get(code)
    if meta:
        result.update(meta)
    return result


def flowering_for_st(st: float | None) -> dict | None:
    """Stade de floraison (61/65/69) pour une ST, ou None si hors floraison.

    Chez l'orge la floraison est parallèle à l'épiaison : à partir de 1450 °C
    les deux coexistent (ex : ST 1500 → épiaison 56 + floraison 65).
    """
    if st is None or not isinstance(st, (int, float)) or st < 1450:
        return None
    code, label = FLOWERING_STAGES[0][1], FLOWERING_STAGES[0][2]
    threshold_hit = FLOWERING_STAGES[0][0]
    for threshold, c, lab in FLOWERING_STAGES:
        if st >= threshold:
            code, label, threshold_hit = c, lab, threshold
        else:
            break
    return {
        "code": code,
        "label": label,
        "principal": 6,
        "sub": int(code[1]) if len(code) == 2 and code.isdigit() else None,
        "st": st,
        "stThreshold": threshold_hit,
        "stBeyond": round((st - threshold_hit) * 10) / 10,
        **FLOWERING_META.get(code, {}),
    }


def stage_since_sowing(daily_values: list[dict], sowing_date_iso: str | None) -> dict:
    """ST + stade Zadoks calculés depuis la date de semis (YYYY-MM-DD).

    daily_values : relevés Open-Meteo {date, tmax, tmin} ou {date, dj}.
    Seuls les jours >= date de semis sont cumulés.
    Inclut la floraison parallèle (clé "flowering", None si hors floraison).
    """
    if not sowing_date_iso:
        return {**stage_for_st(None), "flowering": None, "sowingDate": None}
    relevant = [d for d in daily_values if isinstance(d.get("date"), str) and d["date"] >= sowing_date_iso]
    st = thermal_sum(relevant)
    result = stage_for_st(st)
    result["flowering"] = flowering_for_st(st)
    result["stage7Reached"] = st >= STAGE7_START
    result["sowingDate"] = sowing_date_iso
    result["daysCounted"] = len(relevant)
    return result
