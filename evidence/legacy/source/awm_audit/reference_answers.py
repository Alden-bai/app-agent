"""Deterministic response assembly from real tool outputs only; no DB or verifier access."""


def compute(recipe, results):
    if recipe['$compute'] == 'parts_stock':
        parts = results[0]['parts']
        fits = results[1]['fitment_applications']
        stocks = results[2]['stock_records']
        eligible = {r['part_id'] for r in fits if r['position_id'] == recipe['position_id']}
        return {'parts': [{'part_number': p['part_number'], 'short_description': p['short_description'],
                           'list_price': p['list_price'],
                           'total_on_hand': sum(s['on_hand'] for s in stocks if s['part_id'] == p['id'])}
                          for p in parts if p['id'] in eligible]}
    if recipe['$compute'] == 'protein_bars':
        excluded = {v.casefold() for v in recipe['excluded']}
        output = []
        for index in recipe['indices']:
            s = results[index]
            if 'vegan' not in {v.casefold() for v in s['dietary_tag_codes']}:
                continue
            if excluded.intersection(v.casefold() for v in s['allergen_names']):
                continue
            if recipe.get('no_artificial_sweeteners') and any(
                    term in ingredient.casefold() for ingredient in s['ingredient_names']
                    for term in ('sucralose', 'aspartame', 'acesulfame', 'saccharin')):
                continue
            if sum(r['quantity_on_hand'] for r in s['stock_by_warehouse'] if r['warehouse_code'] == recipe['warehouse']) <= 0:
                continue
            output.append({'sku_id': s['id'], 'sku_code': s['sku_code']})
        return {'skus': sorted(output, key=lambda r: r['sku_id'])}
    raise ValueError('Unknown deterministic answer recipe')
