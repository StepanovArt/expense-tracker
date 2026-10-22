"""Receipt *content* for the synthetic generator: what is printed, and the exact ground truth.

Everything that ends up in the gold JSON is decided here and printed verbatim by the renderer,
so ground truth is exact by construction. Arithmetic is always consistent:
``Σ item totals = subtotal`` and ``subtotal + tax = total`` (CLAUDE.md §4.4). Fields that are
not printed (a receipt without a time line, a restaurant line without unit price) are null in
the gold JSON, matching the prompt rule "null for anything not printed".
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from receipt_vlm.schema import Item, Receipt

Lang = Literal["en", "ar", "ru"]
Layout = Literal["supermarket", "restaurant", "pharmacy"]

# (name, min price, max price, sold by weight) per language and layout. Prices are in a
# "base unit" that is scaled per currency (RUB x ~90).
CATALOG: dict[str, dict[str, list[tuple[str, float, float, bool]]]] = {
    "en": {
        "supermarket": [
            ("Whole Milk 1gal", 3.2, 5.5, False),
            ("Sourdough Bread", 2.5, 6.0, False),
            ("Free Range Eggs 12", 3.5, 7.0, False),
            ("Bananas", 0.5, 0.9, True),
            ("Gala Apples", 1.5, 3.0, True),
            ("Cheddar Cheese", 3.0, 8.0, False),
            ("Greek Yogurt", 1.0, 5.5, False),
            ("Chicken Breast", 5.0, 9.0, True),
            ("Ground Beef 80/20", 4.5, 8.0, False),
            ("Basmati Rice 2lb", 3.0, 6.0, False),
            ("Spaghetti", 1.0, 2.5, False),
            ("Olive Oil 500ml", 6.0, 12.0, False),
            ("Orange Juice", 2.5, 5.0, False),
            ("Coffee Beans", 7.0, 14.0, False),
            ("Green Tea 20ct", 2.0, 4.5, False),
            ("Tomatoes", 1.5, 3.5, True),
            ("Cucumber", 0.6, 1.2, False),
            ("Potatoes 5lb", 3.0, 5.0, False),
            ("Butter Salted", 3.0, 6.0, False),
            ("Corn Flakes", 3.0, 5.5, False),
            ("Peanut Butter", 2.5, 5.0, False),
            ("Sparkling Water", 0.9, 2.0, False),
            ("Dark Chocolate", 2.0, 4.0, False),
            ("Strawberries", 2.5, 5.0, False),
            ("Avocado", 0.8, 2.0, False),
        ],
        "restaurant": [
            ("Cheeseburger", 8.0, 15.0, False),
            ("Chicken Wings", 9.0, 16.0, False),
            ("Caesar Salad", 7.0, 13.0, False),
            ("Margherita Pizza", 10.0, 18.0, False),
            ("Fish & Chips", 11.0, 19.0, False),
            ("Club Sandwich", 8.0, 14.0, False),
            ("French Fries", 3.0, 6.0, False),
            ("Onion Rings", 4.0, 7.0, False),
            ("Grilled Salmon", 16.0, 28.0, False),
            ("Ribeye Steak", 22.0, 42.0, False),
            ("Pasta Carbonara", 12.0, 20.0, False),
            ("Iced Tea", 2.0, 4.0, False),
            ("Lemonade", 2.5, 4.5, False),
            ("Cappuccino", 3.0, 5.5, False),
            ("Latte", 3.5, 6.0, False),
            ("Espresso", 2.0, 3.5, False),
            ("Coca-Cola", 1.5, 3.5, False),
            ("Draft Beer", 5.0, 9.0, False),
            ("House Wine", 7.0, 12.0, False),
            ("Cheesecake", 5.0, 9.0, False),
            ("Brownie Sundae", 6.0, 9.0, False),
        ],
        "pharmacy": [
            ("Ibuprofen 200mg", 4.0, 10.0, False),
            ("Acetaminophen 500mg", 4.0, 9.0, False),
            ("Vitamin C 1000mg", 6.0, 14.0, False),
            ("Multivitamin", 8.0, 20.0, False),
            ("Cough Syrup", 6.0, 12.0, False),
            ("Bandages 30ct", 3.0, 6.0, False),
            ("Hand Sanitizer", 2.0, 5.0, False),
            ("Toothpaste", 2.5, 6.0, False),
            ("Toothbrush", 1.5, 5.0, False),
            ("Shampoo", 4.0, 10.0, False),
            ("Conditioner", 4.0, 10.0, False),
            ("Body Lotion", 5.0, 12.0, False),
            ("Sunscreen SPF50", 8.0, 16.0, False),
            ("Allergy Relief", 7.0, 15.0, False),
            ("Eye Drops", 5.0, 11.0, False),
            ("Cotton Swabs", 1.5, 4.0, False),
            ("Face Masks 10ct", 3.0, 8.0, False),
            ("Thermometer", 8.0, 20.0, False),
            ("Antacid Tablets", 4.0, 8.0, False),
            ("Lip Balm", 1.5, 4.0, False),
        ],
    },
    "ar": {
        "supermarket": [
            ("حليب طازج", 3.0, 7.0, False),
            ("خبز عربي", 1.0, 4.0, False),
            ("أرز بسمتي", 8.0, 25.0, False),
            ("زيت زيتون", 15.0, 40.0, False),
            ("تمر خلاص", 10.0, 30.0, True),
            ("لبن", 2.0, 6.0, False),
            ("جبنة بيضاء", 6.0, 18.0, False),
            ("دجاج طازج", 12.0, 25.0, True),
            ("طماطم", 3.0, 8.0, True),
            ("خيار", 3.0, 7.0, True),
            ("بيض", 8.0, 16.0, False),
            ("ماء معدني", 1.0, 3.0, False),
            ("عصير برتقال", 4.0, 10.0, False),
            ("قهوة عربية", 15.0, 45.0, False),
            ("شاي أحمر", 6.0, 18.0, False),
            ("سكر", 4.0, 9.0, False),
            ("معكرونة", 2.0, 6.0, False),
            ("تفاح أحمر", 5.0, 12.0, True),
            ("موز", 4.0, 8.0, True),
            ("بطاطس", 2.0, 6.0, True),
            ("لحم غنم", 30.0, 60.0, True),
            ("زبادي", 2.0, 6.0, False),
            ("عسل طبيعي", 25.0, 80.0, False),
            ("حمص", 3.0, 8.0, False),
            ("فول مدمس", 2.0, 6.0, False),
        ],
        "restaurant": [
            ("شاورما دجاج", 8.0, 18.0, False),
            ("فلافل", 4.0, 10.0, False),
            ("حمص بالطحينة", 8.0, 16.0, False),
            ("تبولة", 10.0, 20.0, False),
            ("فتوش", 10.0, 20.0, False),
            ("كباب لحم", 25.0, 45.0, False),
            ("مشاوي مشكلة", 40.0, 80.0, False),
            ("برياني دجاج", 20.0, 35.0, False),
            ("مندي لحم", 35.0, 60.0, False),
            ("ليمون بالنعناع", 8.0, 15.0, False),
            ("شاي كرك", 1.0, 5.0, False),
            ("قهوة تركية", 6.0, 14.0, False),
            ("كنافة", 12.0, 25.0, False),
            ("أم علي", 12.0, 22.0, False),
            ("ماء", 1.0, 3.0, False),
            ("مشروب غازي", 3.0, 7.0, False),
            ("سلطة خضراء", 10.0, 18.0, False),
            ("بطاطا مقلية", 6.0, 12.0, False),
            ("متبل", 8.0, 15.0, False),
            ("عصير مانجو", 8.0, 16.0, False),
        ],
        "pharmacy": [
            ("بانادول أقراص", 8.0, 20.0, False),
            ("فيتامين سي", 15.0, 45.0, False),
            ("شراب للكحة", 12.0, 30.0, False),
            ("كريم مرطب", 15.0, 60.0, False),
            ("معجون أسنان", 6.0, 18.0, False),
            ("فرشاة أسنان", 5.0, 15.0, False),
            ("شامبو", 12.0, 35.0, False),
            ("صابون", 3.0, 10.0, False),
            ("مناديل ورقية", 3.0, 12.0, False),
            ("ضمادات", 6.0, 18.0, False),
            ("معقم يدين", 6.0, 20.0, False),
            ("كمامات طبية", 8.0, 25.0, False),
            ("قطرة عين", 15.0, 40.0, False),
            ("مرهم", 10.0, 35.0, False),
            ("حفاضات أطفال", 35.0, 90.0, False),
            ("واقي شمس", 40.0, 120.0, False),
            ("أوميغا 3", 30.0, 90.0, False),
            ("ميزان حرارة", 20.0, 70.0, False),
            ("شاش طبي", 4.0, 12.0, False),
            ("بخاخ أنف", 15.0, 35.0, False),
        ],
    },
    "ru": {
        "supermarket": [
            ("Молоко 3,2% 1л", 0.9, 1.4, False),
            ("Хлеб Бородинский", 0.5, 0.9, False),
            ("Сыр Российский", 2.5, 6.0, True),
            ("Масло сливочное", 1.5, 3.0, False),
            ("Яйца С1 10шт", 1.0, 1.8, False),
            ("Гречка 900г", 0.8, 1.5, False),
            ("Сахар 1кг", 0.7, 1.1, False),
            ("Кефир 1%", 0.8, 1.2, False),
            ("Картофель", 0.3, 0.6, True),
            ("Яблоки Голден", 1.2, 2.0, True),
            ("Бананы", 1.0, 1.6, True),
            ("Курица охл.", 2.5, 4.5, True),
            ("Колбаса докторская", 3.0, 6.0, True),
            ("Чай черный 100п", 1.5, 4.0, False),
            ("Кофе молотый", 3.0, 7.0, False),
            ("Макароны 450г", 0.6, 1.5, False),
            ("Рис круглозерный", 0.8, 1.6, False),
            ("Сметана 20%", 0.8, 1.5, False),
            ("Творог 5%", 1.0, 2.0, False),
            ("Вода питьевая 1,5л", 0.4, 0.9, False),
            ("Сок апельсиновый", 1.2, 2.5, False),
            ("Помидоры", 1.5, 3.5, True),
            ("Огурцы", 1.2, 3.0, True),
            ("Шоколад молочный", 0.9, 2.0, False),
            ("Печенье овсяное", 0.8, 1.6, False),
        ],
        "restaurant": [
            ("Борщ", 3.5, 6.0, False),
            ("Пельмени", 4.0, 7.0, False),
            ("Цезарь с курицей", 5.0, 8.0, False),
            ("Блины со сметаной", 3.0, 5.0, False),
            ("Солянка", 4.0, 6.5, False),
            ("Шашлык из свинины", 7.0, 12.0, False),
            ("Морс клюквенный", 1.5, 3.0, False),
            ("Чай с лимоном", 1.5, 3.0, False),
            ("Капучино", 2.0, 3.5, False),
            ("Эспрессо", 1.5, 2.5, False),
            ("Пирожок с капустой", 0.8, 1.5, False),
            ("Салат Оливье", 3.0, 5.0, False),
            ("Котлета по-киевски", 5.0, 8.0, False),
            ("Картофельное пюре", 1.5, 2.5, False),
            ("Компот", 1.0, 2.0, False),
            ("Хачапури", 4.0, 7.0, False),
            ("Медовик", 2.5, 4.0, False),
            ("Сырники", 3.0, 5.0, False),
            ("Квас", 1.0, 2.0, False),
            ("Вареники с картошкой", 3.0, 5.0, False),
        ],
        "pharmacy": [
            ("Парацетамол 500мг", 0.5, 1.5, False),
            ("Ибупрофен 200мг", 0.8, 2.0, False),
            ("Аскорбиновая кислота", 0.3, 0.9, False),
            ("Активированный уголь", 0.3, 1.0, False),
            ("Пластырь", 0.5, 2.0, False),
            ("Бинт стерильный", 0.5, 1.5, False),
            ("Зубная паста", 1.5, 4.0, False),
            ("Шампунь", 2.5, 6.0, False),
            ("Крем для рук", 1.5, 4.0, False),
            ("Витамин D3", 3.0, 8.0, False),
            ("Капли назальные", 1.5, 4.0, False),
            ("Термометр", 3.0, 9.0, False),
            ("Маска медицинская", 0.3, 1.2, False),
            ("Антисептик", 1.0, 3.0, False),
            ("Валидол", 0.5, 1.5, False),
            ("Но-шпа", 2.0, 5.0, False),
            ("Мирамистин", 3.0, 7.0, False),
            ("Детский крем", 1.0, 2.5, False),
            ("Влажные салфетки", 0.8, 2.0, False),
            ("Сироп от кашля", 2.5, 6.0, False),
        ],
    },
}

MERCHANT_PARTS: dict[str, dict[str, tuple[list[str], list[str]]]] = {
    "en": {
        "supermarket": (
            [
                "FRESHMART",
                "GREEN VALLEY",
                "SUNRISE",
                "CITY FOODS",
                "HARBOR",
                "MAPLE LEAF",
                "BLUE RIVER",
                "GOLDEN FIELD",
                "PARKSIDE",
                "NORTHGATE",
            ],
            ["MARKET", "GROCERY", "SUPERMARKET", "FOODS", "GROCERS"],
        ),
        "restaurant": (
            [
                "THE OAK",
                "BLUE DOOR",
                "MARIO'S",
                "GOLDEN DRAGON",
                "CORNER",
                "RED BARN",
                "SILVER SPOON",
                "OLIVE TREE",
                "LUCKY STAR",
                "HARVEST",
            ],
            ["GRILL", "BISTRO", "PIZZERIA", "KITCHEN", "CAFE", "DINER", "TAVERN"],
        ),
        "pharmacy": (
            [
                "WELLCARE",
                "CITY",
                "HEALTH PLUS",
                "GREENCROSS",
                "FAMILY",
                "MEDIPOINT",
                "CARELINE",
                "VITALITY",
                "SUNSHINE",
                "PRIME",
            ],
            ["PHARMACY", "DRUG STORE", "CHEMIST", "DRUGS", "HEALTH"],
        ),
    },
    "ar": {
        "supermarket": (
            ["أسواق", "سوبرماركت", "بقالة", "هايبر ماركت"],
            [
                "النخيل",
                "الريم",
                "المدينة",
                "السلام",
                "الخير",
                "الفجر",
                "الوادي",
                "الصفا",
                "البركة",
                "الياسمين",
            ],
        ),
        "restaurant": (
            ["مطعم", "كافتيريا", "مشويات", "مطبخ"],
            [
                "الشام",
                "بيت الكرم",
                "الواحة",
                "ليالي بيروت",
                "الأصيل",
                "الضيافة",
                "السرايا",
                "الديوان",
                "الزيتونة",
                "القصر",
            ],
        ),
        "pharmacy": (
            ["صيدلية"],
            [
                "الشفاء",
                "الحياة",
                "النور",
                "الدواء الأولى",
                "الرحمة",
                "البلسم",
                "العافية",
                "ابن سينا",
                "الأمل",
                "الصحة",
            ],
        ),
    },
    "ru": {
        "supermarket": (
            ["Магазин", "Супермаркет", "Продукты", "Минимаркет", "Универсам"],
            [
                "«Берёзка»",
                "«Солнечный»",
                "«Удача»",
                "«Радуга»",
                "«Восход»",
                "«Колос»",
                "«Ромашка»",
                "«Лайм»",
                "«Сосновый бор»",
                "«Весна»",
            ],
        ),
        "restaurant": (
            ["Кафе", "Ресторан", "Столовая", "Кофейня", "Бистро"],
            [
                "«Уют»",
                "«Самовар»",
                "«Пельменная №1»",
                "«Теремок-Сад»",
                "«Матрёшка»",
                "«Белые ночи»",
                "«Старый город»",
                "«Каравай»",
                "«Огонёк»",
                "«Изба»",
            ],
        ),
        "pharmacy": (
            ["Аптека", "Аптечный пункт", "Аптека №12"],
            [
                "«Здоровье»",
                "«Будь здоров»",
                "«Фармация»",
                "«Живика»",
                "«Панацея»",
                "«Доктор»",
                "«Мелодия здоровья»",
                "«Добрая»",
                "«Вита»",
                "«Норма»",
            ],
        ),
    },
}

LABELS: dict[str, dict[str, list[str]]] = {
    "en": {
        "subtotal": ["SUBTOTAL", "Sub Total", "SUB-TOTAL"],
        "tax": ["TAX {rate}%", "VAT {rate}%", "Sales Tax {rate}%"],
        "total": ["TOTAL", "TOTAL DUE", "AMOUNT DUE", "BALANCE DUE"],
        "date": ["Date:", "DATE", ""],
        "time": ["Time:", "TIME", ""],
        "cashier": ["Cashier: {name}", "Server: {name}", "Clerk: {name}"],
        "receipt_no": ["Receipt #{n}", "Check #{n}", "Trans: {n}", "Order #{n}"],
        "footer": [
            "THANK YOU!",
            "Thank you for shopping with us",
            "Please come again",
            "Have a nice day!",
            "Returns accepted within 30 days",
        ],
        "header": ["", "SALES RECEIPT", "CUSTOMER COPY", "TAX INVOICE"],
        "qty_hdr": ["QTY ITEM", "Qty Description"],
        "amount_hdr": ["AMOUNT", "PRICE"],
        "change": ["CHANGE", "Change Due"],
        "tendered": ["CASH TENDERED", "Cash"],
    },
    "ar": {
        "subtotal": ["المجموع الفرعي", "الإجمالي قبل الضريبة"],
        "tax": ["ضريبة القيمة المضافة {rate}%", "الضريبة {rate}%"],
        "total": ["الإجمالي", "المجموع", "المبلغ المستحق", "الإجمالي شامل الضريبة"],
        "date": ["التاريخ:", "التاريخ"],
        "time": ["الوقت:", "الوقت"],
        "cashier": ["الكاشير: {name}", "المحاسب: {name}"],
        "receipt_no": ["رقم الفاتورة: {n}", "فاتورة رقم {n}"],
        "footer": [
            "شكراً لزيارتكم",
            "شكراً لتسوقكم معنا",
            "نتمنى لكم يوماً سعيداً",
            "البضاعة المباعة لا ترد ولا تستبدل",
        ],
        "header": ["فاتورة ضريبية", "فاتورة ضريبية مبسطة", ""],
        "qty_hdr": ["الصنف"],
        "amount_hdr": ["المبلغ"],
        "change": ["الباقي"],
        "tendered": ["المدفوع"],
    },
    "ru": {
        "subtotal": ["Подытог", "Сумма без НДС", "СУММА"],
        "tax": ["НДС {rate}%", "Сумма НДС {rate}%", "НДС {rate}% сумма"],
        "total": ["ИТОГ", "ИТОГО", "К ОПЛАТЕ", "ИТОГО К ОПЛАТЕ"],
        "date": ["Дата:", "ДАТА", ""],
        "time": ["Время:", "ВРЕМЯ", ""],
        "cashier": ["Кассир: {name}", "Кассир {name}", "Продавец: {name}"],
        "receipt_no": ["Чек № {n}", "ЧЕК {n}", "Смена 42 Чек {n}"],
        "footer": [
            "Спасибо за покупку!",
            "СПАСИБО ЗА ПОКУПКУ",
            "Ждём вас снова!",
            "Приятного аппетита!",
        ],
        "header": ["КАССОВЫЙ ЧЕК", "КАССОВЫЙ ЧЕК / ПРИХОД", "Товарный чек"],
        "qty_hdr": ["Наименование", "Товар"],
        "amount_hdr": ["Сумма"],
        "change": ["СДАЧА", "Сдача"],
        "tendered": ["ПОЛУЧЕНО", "Наличными"],
    },
}

# printed payment word -> canonical payment_method
PAYMENTS: dict[str, list[tuple[str, str]]] = {
    "en": [
        ("CASH", "cash"),
        ("VISA ****{d4}", "card"),
        ("MASTERCARD ****{d4}", "card"),
        ("AMEX ****{d4}", "card"),
        ("DEBIT CARD", "card"),
        ("APPLE PAY", "mobile"),
        ("GOOGLE PAY", "mobile"),
        ("GIFT CARD", "voucher"),
    ],
    "ar": [
        ("نقداً", "cash"),
        ("نقدي", "cash"),
        ("بطاقة ائتمان", "card"),
        ("بطاقة مدى", "card"),
        ("فيزا ****{d4}", "card"),
        ("آبل باي", "mobile"),
        ("قسيمة شراء", "voucher"),
    ],
    "ru": [
        ("Наличными", "cash"),
        ("НАЛИЧНЫЕ", "cash"),
        ("Безналичными", "card"),
        ("Карта МИР ****{d4}", "card"),
        ("Карта VISA ****{d4}", "card"),
        ("СБП", "mobile"),
        ("Подарочный сертификат", "voucher"),
    ],
}
PAYMENT_LABEL = {
    "en": ["PAYMENT:", "Paid by", "TENDER"],
    "ar": ["طريقة الدفع:", "الدفع:"],
    "ru": ["Оплата:", "Форма оплаты:", "Способ оплаты:"],
}

CASHIERS = {
    "en": ["Anna", "Mike", "Jessica", "David", "Maria", "John", "Sarah", "Kevin"],
    "ar": ["أحمد", "محمد", "فاطمة", "علي", "سارة", "خالد", "مريم", "يوسف"],
    "ru": ["Иванова А.", "Петров С.", "Смирнова Е.", "Кузнецов Д.", "Попова О.", "Соколов И."],
}

ADDRESSES = {
    "en": [
        "{n} Main St, Springfield",
        "{n} Oak Avenue, Riverside",
        "{n} Market Street",
        "{n} King Road, Dubai Marina",
        "{n} High Street, London",
        "{n} Harbour Rd, Dublin",
        "{n} Elm Blvd, Austin TX",
        "Unit {n}, Mall of the Emirates",
    ],
    "ar": [
        "شارع الشيخ زايد، دبي",
        "شارع حمدان، أبوظبي",
        "شارع الوحدة، الشارقة",
        "مردف، دبي",
        "الخالدية، أبوظبي",
        "شارع الكورنيش، عجمان",
        "النهدة، الشارقة",
    ],
    "ru": [
        "г. Москва, ул. Ленина, д. {n}",
        "г. Казань, пр. Победы, д. {n}",
        "г. Санкт-Петербург, Невский пр., д. {n}",
        "г. Новосибирск, ул. Мира, {n}",
        "г. Екатеринбург, ул. Малышева, {n}",
        "г. Самара, ул. Садовая, д. {n}",
    ],
}
TAX_IDS = {
    "en": ["TRN: 100{n9}", "Tax ID {n9}", "VAT No. GB{n9}", ""],
    "ar": ["الرقم الضريبي: 100{n9}", "TRN 100{n9}"],
    "ru": ["ИНН 77{n9}", "ИНН 16{n9}", "ООО «Торговый дом» ИНН 50{n9}"],
}

CURRENCY_BY_LANG: dict[str, list[tuple[str, float]]] = {
    # currency -> price multiplier relative to the catalogue's base unit
    "en": [("USD", 1.0), ("EUR", 1.0), ("AED", 3.6)],
    "ar": [("AED", 1.0)],
    "ru": [("RUB", 90.0), ("EUR", 1.0)],
}
TAX_RATES = {
    "USD": [6.25, 7.0, 8.0, 8.875],
    "EUR": [20.0, 21.0, 23.0],
    "AED": [5.0],
    "RUB": [20.0, 10.0],
}
CURRENCY_TOKENS = {
    "USD": [("$", "prefix")],
    "EUR": [("€", "prefix"), ("€", "suffix"), ("EUR", "prefix"), ("EUR", "suffix")],
    "AED": [("AED", "prefix"), ("AED", "suffix"), ("د.إ", "suffix"), ("درهم", "suffix")],
    "RUB": [("₽", "suffix"), ("руб.", "suffix"), ("RUB", "suffix")],
}
RU_MONTHS = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]
EN_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ARABIC_INDIC = str.maketrans("0123456789.", "٠١٢٣٤٥٦٧٨٩٫")


@dataclass
class NumberStyle:
    decimal: str = "."
    thousands: str = ""
    arabic_digits: bool = False
    currency_token: str = "$"
    currency_pos: Literal["prefix", "suffix"] = "prefix"

    def amount(self, x: float, with_currency: bool = False, digits: int = 2) -> str:
        neg = x < 0
        s = f"{abs(x):,.{digits}f}"  # 1,234.50
        whole, _, frac = s.partition(".")
        whole = whole.replace(",", self.thousands)
        out = whole + (self.decimal + frac if frac else "")
        if self.arabic_digits:
            out = out.translate(ARABIC_INDIC)
        if neg:
            out = "-" + out
        if with_currency:
            out = (
                f"{self.currency_token} {out}"
                if self.currency_pos == "prefix" and len(self.currency_token) > 1
                else (
                    f"{self.currency_token}{out}"
                    if self.currency_pos == "prefix"
                    else f"{out} {self.currency_token}"
                )
            )
        return out

    def number(self, s: str) -> str:
        return s.translate(ARABIC_INDIC) if self.arabic_digits else s


@dataclass
class ItemLine:
    name: str
    quantity: float
    unit_price: float
    total: float
    by_weight: bool
    print_unit_price: bool
    print_quantity: bool


@dataclass
class ReceiptSpec:
    lang: Lang
    layout: Layout
    currency: str
    merchant: str
    address: str
    tax_id: str
    header: str
    date: dt.date
    date_text: str
    time_text: str | None  # None = not printed
    time_value: str
    cashier: str
    receipt_no: str
    items: list[ItemLine]
    subtotal: float
    tax: float
    tax_rate: float
    total: float
    print_subtotal: bool
    print_tax: bool
    payment_text: str | None
    payment_method: str | None
    footer: str
    style: NumberStyle
    labels: dict[str, str] = field(default_factory=dict)
    tendered: float | None = None

    def gold(self) -> Receipt:
        items = [
            Item(
                name=it.name,
                quantity=it.quantity if it.print_quantity else None,
                unit_price=it.unit_price if it.print_unit_price else None,
                total=it.total,
            )
            for it in self.items
        ]
        return Receipt(
            merchant=self.merchant,
            date=self.date.isoformat(),
            time=self.time_value if self.time_text is not None else None,
            currency=self.currency,
            items=items,
            subtotal=self.subtotal if self.print_subtotal else None,
            tax=self.tax if self.print_tax else None,
            total=self.total,
            payment_method=self.payment_method,
            language=self.lang,
        )


def _choice(rng: np.random.Generator, seq: list[str]) -> str:
    return seq[int(rng.integers(len(seq)))]


def _round_price(x: float, currency: str, rng: np.random.Generator) -> float:
    if currency == "RUB":
        return float(round(x)) if rng.random() < 0.7 else round(x, 2)
    cents = round(x, 2)
    if rng.random() < 0.5:  # psychological pricing .99/.49/.95
        cents = float(int(x)) + float(_choice(rng, ["0.99", "0.49", "0.95", "0.50", "0.25"]))
    return max(cents, 0.25)


def _date_text(
    d: dt.date, lang: str, currency: str, style: NumberStyle, rng: np.random.Generator
) -> str:
    if lang == "ru":
        fmts = [f"{d:%d.%m.%Y}", f"{d:%d.%m.%y}", f"{d.day:02d} {RU_MONTHS[d.month - 1]} {d.year}"]
    elif lang == "ar":
        fmts = [f"{d:%d/%m/%Y}", f"{d:%Y/%m/%d}", f"{d:%Y-%m-%d}"]
    elif currency == "USD":  # US receipts: month first
        fmts = [f"{d:%m/%d/%Y}", f"{d:%m/%d/%y}", f"{EN_MONTHS[d.month - 1]} {d.day:02d}, {d.year}"]
    else:
        fmts = [
            f"{d:%d/%m/%Y}",
            f"{d.day:02d}-{EN_MONTHS[d.month - 1]}-{d.year}",
            f"{d:%d.%m.%Y}",
            f"{d:%Y-%m-%d}",
        ]
    return style.number(_choice(rng, fmts))


def make_spec(rng: np.random.Generator, lang: Lang, layout: Layout) -> ReceiptSpec:
    currency, mult = CURRENCY_BY_LANG[lang][int(rng.integers(len(CURRENCY_BY_LANG[lang])))]
    token, pos = CURRENCY_TOKENS[currency][int(rng.integers(len(CURRENCY_TOKENS[currency])))]
    if lang == "ar" and token in ("AED",) and rng.random() < 0.5:
        token, pos = "د.إ", "suffix"
    style = NumberStyle(currency_token=token, currency_pos=pos)  # type: ignore[arg-type]
    if currency == "RUB" or (currency == "EUR" and lang == "ru"):
        style.decimal = _choice(rng, [".", ","])
        style.thousands = _choice(rng, ["", " "])
    elif currency == "EUR" and rng.random() < 0.4:
        style.decimal, style.thousands = ",", "."
    elif rng.random() < 0.3:
        style.thousands = ","
    if lang == "ar" and rng.random() < 0.3:
        style.arabic_digits = True
        style.thousands = ""

    prefixes, names = MERCHANT_PARTS[lang][layout]
    merchant = f"{_choice(rng, prefixes)} {_choice(rng, names)}"

    catalog = CATALOG[lang][layout]
    n_items = int(
        np.clip(rng.poisson({"supermarket": 6, "restaurant": 4, "pharmacy": 3}[layout]) + 1, 1, 14)
    )
    picks = rng.choice(len(catalog), size=min(n_items, len(catalog)), replace=False)
    unit_mode = {"supermarket": 0.6, "restaurant": 0.3, "pharmacy": 0.7}[layout]
    print_unit = bool(rng.random() < unit_mode)
    # restaurants often print a quantity column for every line (including 1)
    qty_column = layout == "restaurant" and rng.random() < 0.6
    items: list[ItemLine] = []
    for idx in picks:
        name, lo, hi, by_weight = catalog[int(idx)]
        price = _round_price(float(rng.uniform(lo, hi)) * mult, currency, rng)
        if by_weight and layout == "supermarket":
            qty = round(float(rng.uniform(0.2, 2.5)), 3)
            total = round(qty * price, 2)
            items.append(ItemLine(name, qty, price, total, True, True, True))
        else:
            qty = float(rng.choice([1, 1, 1, 1, 2, 2, 3, 4]))
            total = round(qty * price, 2)
            # The gold JSON holds exactly what the renderer prints: unit price only on some
            # receipts/lines, quantity whenever it is printed (always when > 1).
            pu = bool(print_unit and (qty > 1 or rng.random() < 0.5))
            pq = pu or qty > 1 or qty_column
            items.append(ItemLine(name, qty, price, total, False, pu, pq))

    subtotal = round(sum(it.total for it in items), 2)
    print_tax = bool(rng.random() < 0.85)
    rate = float(_choice(rng, [str(r) for r in TAX_RATES[currency]])) if print_tax else 0.0
    tax = round(subtotal * rate / 100, 2)
    total = round(subtotal + tax, 2)
    print_subtotal = bool(rng.random() < 0.8) or (print_tax and rng.random() < 0.5)

    d = dt.date(2021, 1, 1) + dt.timedelta(
        days=int(rng.integers(0, (dt.date(2026, 9, 30) - dt.date(2021, 1, 1)).days))
    )
    hh, mm = int(rng.integers(7, 23)), int(rng.integers(0, 60))
    time_value = f"{hh:02d}:{mm:02d}"
    if rng.random() < 0.85:
        if lang == "en" and currency == "USD" and rng.random() < 0.6:
            h12 = hh % 12 or 12
            time_text = f"{h12:02d}:{mm:02d} {'PM' if hh >= 12 else 'AM'}"
        elif rng.random() < 0.3:
            time_text = f"{time_value}:{int(rng.integers(0, 60)):02d}"
        else:
            time_text = time_value
        time_text = style.number(time_text)
    else:
        time_text = None

    payment_text = payment_method = None
    tendered = None
    if rng.random() < 0.85:
        word, payment_method = PAYMENTS[lang][int(rng.integers(len(PAYMENTS[lang])))]
        payment_text = word.replace("{d4}", f"{int(rng.integers(1000, 9999))}")
        if payment_method == "cash" and rng.random() < 0.5:
            step = 100.0 if currency == "RUB" else 5.0
            tendered = float(np.ceil(total / step) * step)
    labels = {k: _choice(rng, v) for k, v in LABELS[lang].items()}
    labels["tax"] = labels["tax"].replace("{rate}", style.number(f"{rate:g}"))
    labels["payment"] = _choice(rng, PAYMENT_LABEL[lang])
    n = int(rng.integers(1, 999))
    return ReceiptSpec(
        lang=lang,
        layout=layout,
        currency=currency,
        merchant=merchant,
        address=style.number(_choice(rng, ADDRESSES[lang]).replace("{n}", str(n))),
        tax_id=style.number(
            _choice(rng, TAX_IDS[lang]).replace("{n9}", f"{int(rng.integers(10**8, 10**9))}")
        ),
        header=labels["header"],
        date=d,
        date_text=_date_text(d, lang, currency, style, rng),
        time_text=time_text,
        time_value=time_value,
        cashier=labels["cashier"].replace("{name}", _choice(rng, CASHIERS[lang])),
        receipt_no=style.number(
            labels["receipt_no"].replace("{n}", str(int(rng.integers(1000, 99999))))
        ),
        items=items,
        subtotal=subtotal,
        tax=tax,
        tax_rate=rate,
        total=total,
        print_subtotal=print_subtotal,
        print_tax=print_tax,
        payment_text=payment_text,
        payment_method=payment_method,
        footer=labels["footer"],
        style=style,
        labels=labels,
        tendered=tendered,
    )
