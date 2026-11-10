from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
from fpdf import FPDF
from fpdf.enums import XPos, YPos

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "samples"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FIXED_DATE = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)


INVOICE = {
    "file": "szamla_2026_0142.pdf",
    "title": "Számla 2026/0142",
    "author": "Napfény Irodaszer Kft.",
    "pages": [
        [
            ("h1", "SZÁMLA"),
            ("p", "Számlaszám: 2026/0142"),
            ("p", "Kiállítás dátuma: 2026.10.15.   Teljesítés dátuma: 2026.10.14.   Fizetési határidő: 2026.10.29."),
            ("p", "Fizetési mód: átutalás"),
            ("h2", "Eladó"),
            ("p", "Napfény Irodaszer Kft.\n1117 Budapest, Kőrösi utca 12.\nAdószám: 12345678-2-43\n"
                  "Bankszámla (IBAN): HU42 1177 3016 1111 1018 0000 0000\n"
                  "E-mail: szamlazas@napfeny.example.com   Telefon: +36 1 234 5678"),
            ("h2", "Vevő"),
            ("p", "Kovács és Társa Bt.\n6720 Szeged, Tisza Lajos körút 41.\nAdószám: 87654321-1-06\n"
                  "Kapcsolattartó: Kovács Péter (kovacs.peter@example.com, +36 30 123 4567)"),
            ("h2", "Tételek"),
            ("table", [
                ["Megnevezés", "Mennyiség", "Egységár (Ft)", "Nettó (Ft)"],
                ["Fénymásolópapír A4, 500 lap", "20 csomag", "2 150", "43 000"],
                ["Golyóstoll, kék", "100 db", "180", "18 000"],
                ["Iratrendező, 8 cm", "25 db", "990", "24 750"],
                ["Tűzőgép, asztali", "5 db", "4 590", "22 950"],
            ]),
            ("p", "Nettó összesen: 108 700 Ft\nÁFA (27%): 29 349 Ft\nFizetendő összesen: 138 049 Ft"),
            ("p", "Kérjük, az átutalás közleményében tüntesse fel a számlaszámot. Késedelmes fizetés esetén "
                  "a Ptk. szerinti késedelmi kamatot számítjuk fel."),
        ]
    ],
}

CONTRACT = {
    "file": "berleti_szerzodes.pdf",
    "title": "Irodahelyiség bérleti szerződés",
    "author": "Duna Ingatlankezelő Zrt.",
    "pages": [
        [
            ("h1", "BÉRLETI SZERZŐDÉS"),
            ("p", "amely létrejött egyrészről a Duna Ingatlankezelő Zrt. (székhely: 1054 Budapest, Szabadság tér 7., "
                  "adószám: 23456789-2-41, képviseli: Szabó Katalin vezérigazgató), mint bérbeadó, másrészről a "
                  "Kovács és Társa Bt. (székhely: 6720 Szeged, Tisza Lajos körút 41., adószám: 87654321-1-06, "
                  "képviseli: Kovács Péter ügyvezető), mint bérlő között az alulírott helyen és napon, az alábbi "
                  "feltételekkel."),
            ("h2", "1. A szerződés tárgya"),
            ("p", "A bérbeadó bérbe adja, a bérlő bérbe veszi a 1054 Budapest, Szabadság tér 7. szám alatti irodaház "
                  "III. emeletén található, 85 négyzetméter alapterületű irodahelyiséget a hozzá tartozó két "
                  "gépkocsibeállóval együtt. A helyiséget a bérlő kizárólag irodai tevékenység céljára használhatja."),
            ("h2", "2. A bérlet időtartama"),
            ("p", "A felek a szerződést határozott időre, 2026. december 1. napjától 2028. november 30. napjáig kötik. "
                  "A bérlő a helyiséget 2026. november 28-án veszi birtokba, az átadás-átvételről jegyzőkönyv készül."),
            ("h2", "3. Bérleti díj és fizetési feltételek"),
            ("p", "A havi bérleti díj 425 000 Ft + ÁFA, amely a közös költséget tartalmazza, a rezsiköltségeket nem. "
                  "A bérleti díjat a bérlő minden hónap 10. napjáig köteles átutalni a bérbeadó HU42 1177 3016 1111 "
                  "1018 0000 0000 számú bankszámlájára. A bérlő a szerződés aláírásakor háromhavi bérleti díjnak "
                  "megfelelő, azaz 1 275 000 Ft óvadékot fizet."),
        ],
        [
            ("h2", "4. A felek jogai és kötelezettségei"),
            ("p", "A bérlő köteles a helyiséget rendeltetésszerűen használni, annak állagát megóvni. Átalakítást csak "
                  "a bérbeadó előzetes írásbeli hozzájárulásával végezhet. A bérbeadó évente egyszer, előzetes "
                  "egyeztetés után jogosult a helyiség állapotát ellenőrizni."),
            ("h2", "5. A szerződés megszűnése"),
            ("p", "A szerződés a határozott idő leteltével megszűnik. Bármelyik fél jogosult a szerződést rendkívüli "
                  "felmondással, azonnali hatállyal megszüntetni, ha a másik fél a szerződésben vállalt lényeges "
                  "kötelezettségét felszólítás ellenére sem teljesíti. A bérleti díj 30 napot meghaladó késedelme "
                  "lényeges szerződésszegésnek minősül."),
            ("h2", "6. Egyéb rendelkezések"),
            ("p", "A szerződéssel kapcsolatos értesítéseket a felek írásban, az alábbi címekre küldik: "
                  "bérbeadó: berles@duna-ingatlan.example.com, bérlő: kovacs.peter@example.com. A szerződésben nem "
                  "szabályozott kérdésekben a Polgári Törvénykönyv rendelkezései irányadók."),
            ("p", "Budapest, 2026. november 5.\n\n"
                  "......................................          ......................................\n"
                  "Duna Ingatlankezelő Zrt., bérbeadó          Kovács és Társa Bt., bérlő"),
        ],
    ],
}

OFFER = {
    "file": "arajanlat_webfejlesztes.pdf",
    "title": "Árajánlat – webáruház fejlesztése",
    "author": "Pixelfa Digitális Kft.",
    "pages": [
        [
            ("h1", "ÁRAJÁNLAT"),
            ("p", "Ajánlat száma: AJ-2026-087\nKelt: Debrecen, 2026. október 22.\nAz ajánlat érvényes: 2026. november 21-ig."),
            ("p", "Címzett: Kovács és Társa Bt., Kovács Péter ügyvezető részére\n"
                  "Ajánlattevő: Pixelfa Digitális Kft., 4025 Debrecen, Piac utca 30.\n"
                  "Kapcsolat: Tóth Gergely projektmenedzser, +36 20 987 6543, toth.gergely@pixelfa.example.com\n"
                  "Weboldal: https://www.pixelfa.example.com"),
            ("h2", "A projekt tárgya"),
            ("p", "Reszponzív webáruház tervezése és fejlesztése irodaszer-termékek online értékesítésére, "
                  "termékkatalógussal, kosárral, bankkártyás fizetéssel és számlázóprogram-integrációval."),
            ("h2", "Költségbecslés"),
            ("table", [
                ["Feladat", "Ráfordítás", "Díj (Ft)"],
                ["Felhasználói felület tervezése", "40 óra", "520 000"],
                ["Frontend fejlesztés", "90 óra", "1 170 000"],
                ["Backend és adatbázis", "110 óra", "1 430 000"],
                ["Fizetési és számlázási integráció", "30 óra", "390 000"],
                ["Tesztelés és élesítés", "25 óra", "325 000"],
            ]),
            ("p", "Összesen: 3 835 000 Ft + ÁFA. A becsült átfutási idő 10 hét, a megrendeléstől számítva. "
                  "Fizetési ütemezés: 30% előleg a szerződéskötéskor, 70% az átadáskor."),
        ]
    ],
}

MINUTES_TXT = """JEGYZŐKÖNYV

Készült: 2026. október 15-én, a Kovács és Társa Bt. szegedi irodájában (6720 Szeged, Tisza Lajos körút 41.)
Tárgy: negyedéves vezetői értekezlet

Jelen vannak:
- Kovács Péter ügyvezető (levezető elnök)
- Nagy Eszter pénzügyi vezető
- Horváth Dániel értékesítési vezető
- Fekete Júlia jegyzőkönyvvezető

1. napirendi pont: a harmadik negyedév pénzügyi eredményei
Nagy Eszter beszámolt arról, hogy a harmadik negyedévben az árbevétel 18 450 000 Ft volt, ami 12%-os
növekedést jelent az előző év azonos időszakához képest. A működési költségek 14 200 000 Ft-ot tettek ki.
A kintlévőségek összege 2 350 000 Ft, ebből 640 000 Ft lejárt határidejű.

2. napirendi pont: új iroda bérlése Budapesten
Kovács Péter ismertette a Duna Ingatlankezelő Zrt. ajánlatát egy 85 négyzetméteres irodára a Szabadság téren.
A havi bérleti díj 425 000 Ft + ÁFA. Az értekezlet egyhangúlag támogatta a szerződés megkötését.
Határidő: a szerződés aláírása 2026. november 5-ig. Felelős: Kovács Péter.

3. napirendi pont: webáruház fejlesztése
Horváth Dániel javasolta, hogy a cég indítson webáruházat. Három árajánlat érkezett, a legkedvezőbb a
Pixelfa Digitális Kft.-é (3 835 000 Ft + ÁFA). Döntés: a további egyeztetésre Horváth Dániel kap megbízást.
Határidő: 2026. november 20. Kérdés esetén elérhető: horvath.daniel@example.com, +36 70 555 1234.

Egyéb: a következő értekezlet időpontja 2027. január 14.

Az értekezlet 11:45-kor véget ért.

Fekete Júlia s.k.                     Kovács Péter s.k.
jegyzőkönyvvezető                     ügyvezető
"""

CALL_TXT = """PÁLYÁZATI FELHÍVÁS

Kisvállalkozások digitális fejlesztésének támogatása – 2026

A pályázat célja a mikro- és kisvállalkozások digitális felkészültségének javítása, különösen
webáruházak, ügyviteli szoftverek és felhőalapú szolgáltatások bevezetésének támogatásával.

Pályázók köre: Magyarországon bejegyzett, legalább két lezárt üzleti évvel rendelkező mikro- és
kisvállalkozások, amelyek éves nettó árbevétele nem haladja meg a 300 000 000 Ft-ot.

A támogatás mértéke: az elszámolható költségek legfeljebb 50%-a, projektenként minimum 1 000 000 Ft,
maximum 5 000 000 Ft. A támogatás vissza nem térítendő.

Elszámolható költségek:
- szoftverfejlesztés és licencdíjak,
- hardverbeszerzés (a támogatás legfeljebb 30%-áig),
- munkatársak képzése,
- tanácsadás és projektmenedzsment.

Benyújtási határidő: 2026. december 15., 16:00 óra.
A pályázatokat elektronikusan kell benyújtani a https://palyazat.example.com oldalon.
Tájékoztatás: palyazat@example.com, +36 1 765 4321 (munkanapokon 9 és 15 óra között).
"""

CUSTOMERS = [
    ["Ügyfélkód", "Név", "Város", "E-mail", "Telefon", "Éves forgalom (Ft)", "Átlagos fizetési késés (nap)"],
    ["U001", "Kovács és Társa Bt.", "Szeged", "kovacs.peter@example.com", "+36 30 123 4567", "4 250 000,50", "3,5"],
    ["U002", "Zöldmező Kertészet Kft.", "Kecskemét", "info@zoldmezo.example.com", "+36 76 456 789", "1 870 000", "0"],
    ["U003", "Balaton Panzió", "Siófok", "foglalas@balatonpanzio.example.com", "+36 84 321 654", "960 000", "12,0"],
    ["U004", "Mátra Autószerviz Kft.", "Gyöngyös", "szerviz@matraauto.example.com", "+36 37 222 333", "2 340 000", "7,25"],
    ["U005", "Tiszta Lap Könyvelőiroda", "Debrecen", "iroda@tisztalap.example.com", "+36 52 444 555", "3 115 000", "1,0"],
    ["U006", "Pécsi Pékség Bt.", "Pécs", "rendeles@pecsipekseg.example.com", "+36 72 111 999", "1 420 000", ""],
]

INVOICES = [
    ["szamlaszam", "kiallitas", "esedekesseg", "vevo", "netto_ft", "afa_ft", "brutto_ft", "statusz"],
    ["2026/0131", "2026-09-02", "2026-09-17", "Zöldmező Kertészet Kft.", "86500", "23355", "109855", "fizetve"],
    ["2026/0135", "2026-09-18", "2026-10-03", "Balaton Panzió", "42000", "11340", "53340", "késedelmes"],
    ["2026/0138", "2026-09-30", "2026-10-15", "Mátra Autószerviz Kft.", "128400", "34668", "163068", "fizetve"],
    ["2026/0140", "2026-10-08", "2026-10-23", "Tiszta Lap Könyvelőiroda", "64250", "17348", "81598", "fizetve"],
    ["2026/0142", "2026-10-15", "2026-10-29", "Kovács és Társa Bt.", "108700", "29349", "138049", "nyitott"],
    ["2026/0145", "2026-10-27", "2026-11-11", "Pécsi Pékség Bt.", "37900", "10233", "48133", "nyitott"],
]


class SamplePdf(FPDF):
    def __init__(self, title: str):
        super().__init__(format="A4")
        self.doc_title = title
        self.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.cell(0, 6, f"{self.doc_title} – {self.page_no()}. oldal", align="C")

    def add_block(self, kind: str, content) -> None:
        if kind == "h1":
            self.set_font("DejaVu", "B", 18)
            self.cell(0, 12, content, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.ln(3)
        elif kind == "h2":
            self.ln(2)
            self.set_font("DejaVu", "B", 12)
            self.cell(0, 8, content, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        elif kind == "p":
            self.set_font("DejaVu", "", 10)
            self.multi_cell(0, 5.5, content, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(2)
        elif kind == "table":
            self.set_font("DejaVu", "", 9)
            with self.table(first_row_as_headings=True, line_height=6.5) as table:
                for row_values in content:
                    row = table.row()
                    for value in row_values:
                        row.cell(value)
            self.ln(3)
        else:
            raise ValueError(f"Ismeretlen blokktípus: {kind}")


def write_pdf(spec: dict, output_dir: Path) -> Path:
    pdf = SamplePdf(spec["title"])
    pdf.set_title(spec["title"])
    pdf.set_author(spec["author"])
    pdf.set_creator("DocProc mintagenerátor")
    pdf.set_creation_date(FIXED_DATE)
    for page in spec["pages"]:
        pdf.add_page()
        for kind, content in page:
            pdf.add_block(kind, content)
    path = output_dir / spec["file"]
    pdf.output(str(path))
    return path


def write_text(path: Path, text: str, encoding: str) -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def write_csv(path: Path, rows: list[list[str]], delimiter: str) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter=delimiter, lineterminator="\n").writerows(rows)
    return path


def generate(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created = [write_pdf(spec, output_dir) for spec in (INVOICE, CONTRACT, OFFER)]
    created.append(write_text(output_dir / "jegyzokonyv_2026_10_15.txt", MINUTES_TXT, "utf-8"))
    created.append(write_text(output_dir / "palyazati_felhivas_cp1250.txt", CALL_TXT, "cp1250"))
    created.append(write_csv(output_dir / "ugyfelek.csv", CUSTOMERS, ";"))
    created.append(write_csv(output_dir / "szamlak_2026.csv", INVOICES, ","))
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Demó dokumentumok (PDF, TXT, CSV) generálása.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="kimeneti mappa")
    args = parser.parse_args()
    for path in generate(args.output):
        print(f"Létrehozva: {path}")


if __name__ == "__main__":
    main()
