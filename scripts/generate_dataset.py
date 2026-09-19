"""Generate a synthetic Moroccan company corpus (no real data) under data/raw.

Company: Maghreb Industries SA, Casablanca. Produces French PDFs (invoices,
policies, contract, procedure), a French text charter and an Arabic HR policy
in Markdown, plus manifest.json describing metadata for scripts/ingest.py.

Arabic is written as Markdown rather than PDF because PDF generation with
proper Arabic shaping needs a shaping-capable font; the ingestion path is
identical for both formats.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from fpdf import FPDF

random.seed(42)
OUT = Path("data/raw")
COMPANY = "Maghreb Industries SA"

SUPPLIERS = [
    ("Atlas Bureautique SARL", "001234567000089", "Casablanca"),
    ("Rif Logistique SA", "002345678000045", "Tanger"),
    ("Souss Informatique SARL", "003456789000012", "Agadir"),
    ("Chaouia Maintenance SARL AU", "004567890000078", "Settat"),
    ("Oriental Energie SA", "005678901000034", "Oujda"),
]
ITEMS = [
    ("Fourniture de chaises de bureau ergonomiques", 1200.0, 20),
    ("Prestation de transport de marchandises Casablanca-Tanger", 8500.0, 20),
    ("Licences logicielles ERP (annuel)", 15000.0, 20),
    ("Maintenance préventive des compresseurs", 6200.0, 20),
    ("Fourniture de gasoil industriel", 9800.0, 10),
    ("Ordinateurs portables professionnels", 7900.0, 20),
    ("Nettoyage industriel des ateliers", 4300.0, 20),
    ("Équipements de protection individuelle", 2150.0, 20),
]


def fmt(amount: float) -> str:
    s = f"{amount:,.2f}".replace(",", " ").replace(".", ",")
    return f"{s} MAD"


class Doc(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 6, COMPANY, align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    pdf = Doc()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, title)
    pdf.ln(3)
    for p in paragraphs:
        if p == "<PAGE>":
            pdf.add_page()
            continue
        if p.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, p[3:])
        else:
            pdf.set_font("Helvetica", size=10.5)
            pdf.multi_cell(0, 5.5, p)
        pdf.ln(2)
    pdf.output(str(path))


def invoices(manifest: list[dict]) -> None:
    start = date(2025, 1, 6)
    for i in range(1, 13):
        supplier, ice, city = SUPPLIERS[(i - 1) % len(SUPPLIERS)]
        desc, unit, tva_rate = ITEMS[(i * 3) % len(ITEMS)]
        qty = random.choice([1, 2, 5, 10, 20])
        ht = round(unit * qty, 2)
        tva = round(ht * tva_rate / 100, 2)
        ttc = round(ht + tva, 2)
        d = start + timedelta(days=21 * i + random.randint(0, 6))
        due = d + timedelta(days=60)
        number = f"F-2025-{i:03d}"
        name = f"facture_{number}.pdf"
        paragraphs = [
            f"Fournisseur : {supplier}",
            f"ICE : {ice}    Ville : {city}",
            f"Client : {COMPANY} - Zone industrielle Ain Sebaa, Casablanca - ICE 000987654000021",
            f"Date de facture : {d.strftime('%d/%m/%Y')}    Échéance : {due.strftime('%d/%m/%Y')}",
            f"Bon de commande : BC-2025-{100 + i}",
            "## Détail",
            f"Désignation : {desc}",
            f"Quantité : {qty}    Prix unitaire HT : {fmt(unit)}",
            f"Montant HT : {fmt(ht)}",
            f"TVA {tva_rate}% : {fmt(tva)}",
            f"Montant TTC : {fmt(ttc)}",
            "## Conditions",
            "Paiement à 60 jours fin de mois par virement bancaire. "
            "Pénalités de retard : 1,5% par mois de retard. Escompte pour paiement anticipé : néant.",
            f"Arrêtée la présente facture à la somme de {fmt(ttc)} toutes taxes comprises.",
        ]
        write_pdf(OUT / name, f"FACTURE N° {number}", paragraphs)
        manifest.append(
            {
                "file": name,
                "department": "finance",
                "doc_type": "invoice",
                "language": "fr",
                "doc_date": d.isoformat(),
                "tags": ["fournisseur", supplier.split()[0].lower(), f"tva{tva_rate}"],
            }
        )


def policies(manifest: list[dict]) -> None:
    write_pdf(
        OUT / "politique_conges.pdf",
        "Politique de congés et absences",
        [
            "## Article 1 - Droit aux congés annuels",
            "Chaque salarié bénéficie de 22 jours ouvrables de congé annuel payé après une année de service continu, "
            "conformément au Code du travail marocain. Les salariés de moins de 18 ans bénéficient de 2 jours supplémentaires par mois de service.",
            "## Article 2 - Demande de congé",
            "La demande de congé doit être déposée au moins 15 jours calendaires avant la date de départ via le portail RH. "
            "Le responsable hiérarchique dispose de 5 jours ouvrables pour valider ou refuser la demande.",
            "## Article 3 - Report des congés",
            "Le report des congés non pris sur l'année suivante est limité à 5 jours ouvrables et doit être approuvé par la Direction des Ressources Humaines avant le 31 décembre.",
            "<PAGE>",
            "## Article 4 - Congés exceptionnels",
            "Mariage du salarié : 4 jours. Naissance d'un enfant : 3 jours. Décès du conjoint, d'un enfant ou d'un parent : 3 jours. "
            "Circoncision : 2 jours. Ces congés sont rémunérés et ne sont pas déduits du congé annuel.",
            "## Article 5 - Absences maladie",
            "Toute absence pour maladie doit être justifiée par un certificat médical transmis à la DRH dans les 48 heures. "
            "Les indemnités journalières sont versées par la CNSS à partir du 4ème jour d'arrêt.",
            "## Article 6 - Jours fériés",
            "Les jours fériés légaux au Maroc sont chômés et payés. Lorsqu'un jour férié tombe un dimanche, aucun report n'est accordé.",
        ],
    )
    manifest.append(
        {"file": "politique_conges.pdf", "department": "rh", "doc_type": "policy", "language": "fr", "doc_date": "2024-01-15", "tags": ["conges", "absences"]}
    )

    write_pdf(
        OUT / "politique_teletravail.pdf",
        "Charte du télétravail",
        [
            "## 1. Éligibilité",
            "Le télétravail est ouvert aux salariés en CDI ayant au moins 6 mois d'ancienneté et dont les fonctions sont compatibles avec le travail à distance.",
            "## 2. Rythme",
            "Le télétravail est limité à 2 jours par semaine, fixés d'un commun accord avec le manager. Le lundi est une journée de présence obligatoire sur site.",
            "## 3. Équipement et indemnité",
            "L'entreprise fournit un ordinateur portable et un accès VPN. Une indemnité forfaitaire de 300 MAD par mois couvre les frais de connexion et d'électricité.",
            "## 4. Horaires et joignabilité",
            "Le salarié en télétravail doit être joignable entre 9h00 et 12h30 et entre 14h00 et 17h30. "
            "Toute réunion d'équipe se tient en visioconférence sur Microsoft Teams.",
        ],
    )
    manifest.append(
        {"file": "politique_teletravail.pdf", "department": "rh", "doc_type": "policy", "language": "fr", "doc_date": "2024-09-01", "tags": ["teletravail"]}
    )

    (OUT / "politique_conges_ar.md").write_text(
        """# سياسة العطل والغيابات

## المادة 1 - الحق في العطلة السنوية
يستفيد كل أجير من 22 يوم عمل من العطلة السنوية المؤدى عنها بعد سنة من الخدمة المتواصلة، طبقا لمدونة الشغل المغربية. يستفيد الأجراء الذين تقل أعمارهم عن 18 سنة من يومين إضافيين عن كل شهر من الخدمة.

## المادة 2 - طلب العطلة
يجب تقديم طلب العطلة قبل 15 يوما على الأقل من تاريخ المغادرة عبر بوابة الموارد البشرية. يتوفر المسؤول المباشر على 5 أيام عمل للموافقة على الطلب أو رفضه.

## المادة 3 - ترحيل العطل
يقتصر ترحيل العطل غير المستعملة إلى السنة الموالية على 5 أيام عمل ويجب أن توافق عليه مديرية الموارد البشرية قبل 31 دجنبر.

## المادة 4 - العطل الاستثنائية
زواج الأجير: 4 أيام. ازدياد مولود: 3 أيام. وفاة الزوج أو الابن أو أحد الوالدين: 3 أيام. الختان: يومان. هذه العطل مؤدى عنها ولا تخصم من العطلة السنوية.

## المادة 5 - الغياب بسبب المرض
يجب تبرير كل غياب بسبب المرض بشهادة طبية تسلم إلى مديرية الموارد البشرية خلال 48 ساعة. يصرف الصندوق الوطني للضمان الاجتماعي التعويضات اليومية ابتداء من اليوم الرابع من التوقف عن العمل.
""",
        encoding="utf-8",
    )
    manifest.append(
        {"file": "politique_conges_ar.md", "department": "rh", "doc_type": "policy", "language": "ar", "doc_date": "2024-01-15", "tags": ["conges", "absences"]}
    )


def procurement(manifest: list[dict]) -> None:
    write_pdf(
        OUT / "procedure_achats.pdf",
        "Procédure d'achats et d'engagement des dépenses",
        [
            "## 1. Objet",
            "Cette procédure définit les seuils d'approbation et les étapes obligatoires pour tout achat de biens ou de services au sein de l'entreprise.",
            "## 2. Seuils d'approbation",
            "Achats inférieurs à 10 000 MAD HT : validation du responsable de service. "
            "Achats de 10 000 à 50 000 MAD HT : validation du responsable de service et du Directeur Achats. "
            "Achats supérieurs à 50 000 MAD HT : validation du Directeur Achats et du Directeur Général, avec au minimum trois devis comparatifs.",
            "## 3. Bon de commande",
            "Aucune commande ne peut être passée sans bon de commande (BC) émis dans l'ERP. Le numéro de BC doit figurer sur la facture du fournisseur, faute de quoi la facture est retournée.",
            "## 4. Réception et contrôle",
            "La réception est constatée par un procès-verbal signé par le demandeur. Le service comptabilité rapproche la facture, le BC et le PV de réception avant tout paiement (rapprochement à trois voies).",
            "## 5. Délais de paiement",
            "Le délai de paiement standard des fournisseurs est de 60 jours fin de mois. Tout délai inférieur doit être justifié et approuvé par le Directeur Financier.",
        ],
    )
    manifest.append(
        {"file": "procedure_achats.pdf", "department": "achats", "doc_type": "procedure", "language": "fr", "doc_date": "2024-03-10", "tags": ["achats", "seuils"]}
    )

    write_pdf(
        OUT / "contrat_cadre_atlas_bureautique.pdf",
        "Contrat cadre de fourniture - Atlas Bureautique SARL",
        [
            "Entre Maghreb Industries SA, ci-après « le Client », et Atlas Bureautique SARL, ICE 001234567000089, ci-après « le Fournisseur ».",
            "## Article 1 - Objet",
            "Le Fournisseur s'engage à fournir au Client du mobilier et des consommables de bureau selon le catalogue annexé.",
            "## Article 2 - Durée",
            "Le présent contrat est conclu pour une durée de 24 mois à compter du 1er février 2025, renouvelable par tacite reconduction pour des périodes de 12 mois.",
            "## Article 3 - Prix et remise",
            "Les prix du catalogue sont fermes pendant 12 mois. Une remise de 8% est appliquée sur toute commande supérieure à 30 000 MAD HT.",
            "## Article 4 - Livraison",
            "Le délai de livraison maximal est de 10 jours ouvrables à compter de la réception du bon de commande. Tout retard supérieur à 5 jours ouvrables donne lieu à une pénalité de 0,5% du montant de la commande par jour de retard, plafonnée à 10%.",
            "## Article 5 - Résiliation",
            "Chaque partie peut résilier le contrat moyennant un préavis de 3 mois notifié par lettre recommandée avec accusé de réception.",
            "## Article 6 - Droit applicable",
            "Le présent contrat est soumis au droit marocain. Tout litige relève de la compétence du Tribunal de Commerce de Casablanca.",
        ],
    )
    manifest.append(
        {"file": "contrat_cadre_atlas_bureautique.pdf", "department": "achats", "doc_type": "contract", "language": "fr", "doc_date": "2025-02-01", "tags": ["contrat", "atlas"]}
    )


def misc(manifest: list[dict]) -> None:
    (OUT / "charte_securite_si.txt").write_text(
        """CHARTE DE SÉCURITÉ DES SYSTÈMES D'INFORMATION

1. Mots de passe
Les mots de passe doivent comporter au moins 12 caractères, mélanger majuscules, minuscules, chiffres et symboles, et être renouvelés tous les 90 jours. L'authentification à deux facteurs est obligatoire pour l'accès VPN et à la messagerie.

2. Postes de travail
Tout poste doit être verrouillé en cas d'absence. Le chiffrement du disque (BitLocker) est activé sur tous les ordinateurs portables. L'installation de logiciels est réservée au service informatique.

3. Données
Les documents classés « Confidentiel » ne doivent jamais être envoyés à une adresse e-mail externe ni stockés sur un service cloud personnel. Les clés USB personnelles sont interdites.

4. Incidents
Tout incident de sécurité (phishing, perte d'équipement, accès suspect) doit être signalé au service informatique dans l'heure via le numéro interne 4444 ou l'adresse securite@maghreb-industries.example.
""",
        encoding="utf-8",
    )
    manifest.append(
        {"file": "charte_securite_si.txt", "department": "it", "doc_type": "policy", "language": "fr", "doc_date": "2024-06-01", "tags": ["securite"]}
    )

    (OUT / "procedure_notes_de_frais.md").write_text(
        """# Procédure de notes de frais

## Plafonds
- Repas en déplacement : 150 MAD par repas sur justificatif.
- Hôtel au Maroc : 900 MAD par nuit maximum (catégorie 3 ou 4 étoiles).
- Indemnité kilométrique véhicule personnel : 2,5 MAD par kilomètre.

## Délais
Les notes de frais sont saisies dans l'ERP avant le 5 du mois suivant, avec les justificatifs scannés. Le remboursement intervient avec la paie du mois.

## Refus
Sont refusés : les boissons alcoolisées, les amendes, les dépenses sans justificatif et les frais de déplacement domicile-bureau.
""",
        encoding="utf-8",
    )
    manifest.append(
        {"file": "procedure_notes_de_frais.md", "department": "finance", "doc_type": "procedure", "language": "fr", "doc_date": "2024-05-20", "tags": ["frais"]}
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    invoices(manifest)
    policies(manifest)
    procurement(manifest)
    misc(manifest)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} documents in {OUT}/ (manifest.json written)")


if __name__ == "__main__":
    main()
