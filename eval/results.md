# Evaluation (40 questions, top_k=5)

LLM: `ollama:qwen3.5:4b` · embedding dim: 384 · chunking: `recursive`

| Metric | Value |
|---|---|
| retrieval_hit_at_k | 1.0 |
| mrr | 0.944 |
| answered_when_answerable | 1.0 |
| refused_when_unanswerable | 1.0 |
| hallucination_rate | 0.0 |
| keyword_accuracy | 0.941 |
| citation_validity | 0.978 |
| median_latency_ms | 3591 |
| median_cache_latency_ms | 29 |
| cache_hit_rate_on_repeat | 0.85 |

## Per question

| # | Q | answerable | found | hit | grounding | citations |
|---|---|---|---|---|---|---|
| 1 | Combien de jours de congé annuel payé a droit un salarié ? | True | True | True | 1.0 | politique_conges.pdf#p1 |
| 2 | Combien de jours avant le départ faut-il déposer une demande | True | True | True | 1.0 | politique_conges.pdf#p1 |
| 3 | Combien de jours de congé peut-on reporter sur l'année suiva | True | True | True | 1.0 | politique_conges.pdf#p1 |
| 4 | Combien de jours de congé pour le mariage du salarié ? | True | True | True | 1.0 | politique_conges.pdf#p2 |
| 5 | Sous quel délai faut-il transmettre un certificat médical ? | True | True | True | 1.0 | politique_conges.pdf#p2, politique_conges_ar.md#p1 |
| 6 | كم عدد أيام العطلة السنوية المؤدى عنها؟ | True | True | True | 1.0 | politique_conges_ar.md#p1, politique_conges.pdf#p1 |
| 7 | كم يوما من العطلة يمنح عند زواج الأجير؟ | True | True | True | 1.0 | politique_conges_ar.md#p1, politique_conges.pdf#p2 |
| 8 | ما هي مدة الإشعار المسبق لطلب العطلة؟ | True | True | True | 1.0 | politique_conges_ar.md#p1, politique_conges.pdf#p1 |
| 9 | Combien de jours de télétravail par semaine sont autorisés ? | True | True | True | 1.0 | politique_teletravail.pdf#p1 |
| 10 | Quel est le montant de l'indemnité mensuelle de télétravail  | True | True | True | 1.0 | politique_teletravail.pdf#p1 |
| 11 | Quel jour de la semaine la présence sur site est-elle obliga | True | True | True | 1.0 | politique_teletravail.pdf#p1 |
| 12 | Quel est le montant TTC de la facture F-2025-003 ? | True | True | True | 1.0 | facture_F-2025-003.pdf#p1 |
| 13 | Quel est le numéro ICE du fournisseur de la facture F-2025-0 | True | True | True | 1.0 | facture_F-2025-001.pdf#p1 |
| 14 | Quel est le délai de paiement indiqué sur les factures fourn | True | True | True | 1.0 | procedure_achats.pdf#p1, facture_F-2025-004.pdf#p1, facture_F-2025-007.pdf#p1, facture_F-2025-005.pdf#p1 |
| 15 | Quel taux de pénalité de retard de paiement figure sur les f | True | True | True | 1.0 | facture_F-2025-007.pdf#p1, facture_F-2025-004.pdf#p1, facture_F-2025-005.pdf#p1, facture_F-2025-008.pdf#p1 |
| 16 | Quel est le numéro de bon de commande de la facture F-2025-0 | True | True | True | 1.0 | facture_F-2025-007.pdf#p1 |
| 17 | À partir de quel montant faut-il trois devis comparatifs ? | True | True | True | 1.0 | procedure_achats.pdf#p1, procedure_achats.pdf#p1 |
| 18 | Qui valide un achat de 30 000 MAD HT ? | True | True | True | 1.0 | procedure_achats.pdf#p1 |
| 19 | Qu'est-ce que le rapprochement à trois voies ? | True | True | True | 1.0 | procedure_achats.pdf#p1 |
| 20 | Quelle est la durée du contrat cadre avec Atlas Bureautique  | True | True | True | 1.0 | contrat_cadre_atlas_bureautique.pdf#p1, contrat_cadre_atlas_bureautique.pdf#p1 |
| 21 | Quelle remise s'applique sur les commandes supérieures à 30  | True | True | True | 1.0 | contrat_cadre_atlas_bureautique.pdf#p1 |
| 22 | Quel est le préavis de résiliation du contrat Atlas Bureauti | True | True | True | 1.0 | contrat_cadre_atlas_bureautique.pdf#p1 |
| 23 | Quel tribunal est compétent en cas de litige avec Atlas Bure | True | True | True | 1.0 | contrat_cadre_atlas_bureautique.pdf#p1 |
| 24 | Quelle est la longueur minimale d'un mot de passe ? | True | True | True | 1.0 | charte_securite_si.txt#p1 |
| 25 | Tous les combien de jours faut-il changer son mot de passe ? | True | True | True | 1.0 | charte_securite_si.txt#p1 |
| 26 | Quel numéro appeler pour signaler un incident de sécurité ? | True | True | True | 1.0 | charte_securite_si.txt#p1 |
| 27 | Quel est le plafond d'un repas en déplacement ? | True | True | True | 1.0 | procedure_notes_de_frais.md#p1 |
| 28 | Quelle est l'indemnité kilométrique pour un véhicule personn | True | True | True | 1.0 | procedure_notes_de_frais.md#p1 |
| 29 | Avant quelle date du mois doit-on saisir ses notes de frais  | True | True | True | 1.0 | procedure_notes_de_frais.md#p1 |
| 30 | Quel est le plafond d'une nuit d'hôtel au Maroc ? | True | True | True | 1.0 | procedure_notes_de_frais.md#p1 |
| 31 | Quel est le chiffre d'affaires de Maghreb Industries en 2024 | False | False | False | 1.0 |  |
| 32 | Qui est le directeur général de l'entreprise ? | False | False | False | 1.0 |  |
| 33 | Quelle est la politique de prime de fin d'année ? | False | False | False | 1.0 |  |
| 34 | ما هو راتب المدير المالي؟ | False | False | False | 1.0 |  |
| 35 | Quelle est la capitale de l'Australie ? | False | False | False | 1.0 |  |
| 36 | Combien de jours de congé pour la naissance d'un enfant ? | True | True | True | 1.0 | politique_conges.pdf#p2 |
| 37 | Quel est le délai de livraison maximal d'Atlas Bureautique ? | True | True | True | 1.0 | contrat_cadre_atlas_bureautique.pdf#p1 |
| 38 | Les clés USB personnelles sont-elles autorisées ? | True | True | True | 1.0 | charte_securite_si.txt#p1 |
| 39 | Quel est le taux de TVA sur la facture F-2025-005 ? | True | True | True | 1.0 | facture_F-2025-003.pdf#p1 |
| 40 | Quelle est la politique de remboursement des frais de taxi à | False | False | False | 1.0 |  |