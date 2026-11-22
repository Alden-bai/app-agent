# AWM 推理技能库 v1

从训练轨迹提炼的规则式工作流；不包含模型权重更新。实际效果尚待评测。

## lesson_protocol_error_limit

适用范围：通用失败提醒

When producing a structured tool action

- Return exactly one valid JSON action with the documented fields. Read the schema and correct the rejected action before trying again; do not repeat malformed output.

来源：51 条训练轨迹；精确哈希和成员路径见 bank.json。

## lesson_stalled

适用范围：通用失败提醒

When discovery or actions repeat without new information

- Review the last observation. Change search terms, follow pagination, or read the specific schema needed for the next step. Do not repeat an identical request without a new reason.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_00cfd5b8ae073ed8

适用范围：dating_3

When the request requires operations involving: update_my_user_preferences

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. update_my_user_preferences: supply current values for age_max, age_min, max_distance_km, require_non_smoker, require_ok_pets, want_children_future. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0181beb8f77023f5

适用范围：communications_messaging_platform_1

When the request requires operations involving: get_messaging_service_by_name_for_user, update_messaging_service_by_sid

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_messaging_service_by_name_for_user: supply current values for name. Inspect the result before proceeding.
- 2. update_messaging_service_by_sid: supply current values for fallback_method, fallback_url, service_sid. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_02556176a39a4b17

适用范围：billing_payments_4

When the request requires operations involving: get_subscription, create_invoice, create_invoice_line_item, finalize_invoice

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_subscription: supply current values for subscription_id. Inspect the result before proceeding.
- 2. create_invoice: supply current values for currency, customer_id, subscription_id. Inspect the result before proceeding.
- 3. create_invoice_line_item: supply current values for amount, description, invoice_id, quantity, tax_rate, unit_amount. Inspect the result before proceeding.
- 4. finalize_invoice: supply current values for invoice_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_02f61373290be2ef

适用范围：social_network_2

When the request requires operations involving: get_guild_by_name, get_guild_channel_by_name, create_channel_post, get_guild_role_by_name, create_post_comment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_guild_by_name: supply current values for guild_name. Inspect the result before proceeding.
- 2. get_guild_channel_by_name: supply current values for channel_name, guild_id. Inspect the result before proceeding.
- 3. create_channel_post: supply current values for channel_id, content. Inspect the result before proceeding.
- 4. get_guild_role_by_name: supply current values for guild_id, role_name. Inspect the result before proceeding.
- 5. create_post_comment: supply current values for content, post_id, tagged_role_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_03ac612b8ff4a25d

适用范围：e_commerce_ordering_4

When the request requires operations involving: list_my_restaurants, get_my_favorite_menu_items, list_my_restaurants, get_location

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_my_restaurants: supply current values for no arguments. Inspect the result before proceeding.
- 2. get_my_favorite_menu_items: supply current values for days, limit. Inspect the result before proceeding.
- 3. list_my_restaurants: supply current values for no arguments. Inspect the result before proceeding.
- 4. get_location: supply current values for location_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_050ac0bbeba702f1

适用范围：booking_marketplace_4

When the request requires operations involving: list_unpaid_payouts

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_unpaid_payouts: supply current values for from_date, to_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_06ec03a1e6e4eb12

适用范围：booking_marketplace_1

When the request requires operations involving: search_locations, search_listings, create_instant_booking

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_locations: supply current values for city, country. Inspect the result before proceeding.
- 2. search_listings: supply current values for children_date, city, country, gift_date, limit, sort_by, sort_order. Inspect the result before proceeding.
- 3. create_instant_booking: supply current values for checkin_date, checkout_date, guests_count, listing_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_07dd76546779cd79

适用范围：transportation_logistics_1

When the request requires operations involving: create_ride_request

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_ride_request: supply current values for appointment_location, appointment_time, assistance_required, companions_count, pickup_address, rider_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0ae27ea4f6b06e29

适用范围：property_management_1

When the request requires operations involving: list_property_units, create_maintenance_request, create_maintenance_request_time_slot, create_maintenance_assignment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_property_units: supply current values for property_id. Inspect the result before proceeding.
- 2. create_maintenance_request: supply current values for category, description, priority, property_id, tenant_id, unit_id. Inspect the result before proceeding.
- 3. create_maintenance_request_time_slot: supply current values for end_datetime, request_id, start_datetime. Inspect the result before proceeding.
- 4. create_maintenance_assignment: supply current values for assigned_user_id, request_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0b604ba22c42de7c

适用范围：asset_management_6

When the request requires operations involving: list_assets, list_shops, list_service_reminders, create_work_order, create_work_order_task

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_assets: supply current values for vin. Inspect the result before proceeding.
- 2. list_shops: supply current values for name, type. Inspect the result before proceeding.
- 3. list_service_reminders: supply current values for asset_id. Inspect the result before proceeding.
- 4. create_work_order: supply current values for asset_id, description, estimated_total_cost, opened_at, service_reminder_id, shop_id, status, title. Inspect the result before proceeding.
- 5. create_work_order_task: supply current values for description, estimated_cost, position, work_order_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0b932852969d130d

适用范围：inventory_management_1

When the request requires operations involving: get_reorder_rule_by_location_and_sku, list_reorder_rules, patch_reorder_rule_by_id, evaluate_reorder_for_location_and_sku

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_reorder_rule_by_location_and_sku: supply current values for internal_sku, location_name. Inspect the result before proceeding.
- 2. list_reorder_rules: supply current values for internal_sku, location_name. Inspect the result before proceeding.
- 3. patch_reorder_rule_by_id: supply current values for max_quantity, min_quantity, preferred_supplier_name, reorder_rule_id. Inspect the result before proceeding.
- 4. evaluate_reorder_for_location_and_sku: supply current values for internal_sku, location_name. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0c772e4bab8dc0cd

适用范围：b2b_saas_4

When the request requires operations involving: get_property_by_address, list_tenants, get_unit_by_property_and_number, create_lease, create_lease_document

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_property_by_address: supply current values for address_line1, city, postal_code, state. Inspect the result before proceeding.
- 2. list_tenants: supply current values for name. Inspect the result before proceeding.
- 3. get_unit_by_property_and_number: supply current values for property_id, unit_number. Inspect the result before proceeding.
- 4. create_lease: supply current values for base_rent_annual, base_rent_currency, billing_schedule, escalation_clauses, lease_end_date, lease_start_date, portfolio_id, property_id, security_deposit, security_deposit_currency, tenant_id, unit_id. Inspect the result before proceeding.
- 5. create_lease_document: supply current values for file_name, lease_id, storage_location. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0daafd969133132c

适用范围：iot_smart_home_control_5

When the request requires operations involving: list_owned_homes, get_room_by_name, list_home_members, update_home_member

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_owned_homes: supply current values for no arguments. Inspect the result before proceeding.
- 2. get_room_by_name: supply current values for home_id, room_name. Inspect the result before proceeding.
- 3. list_home_members: supply current values for home_id. Inspect the result before proceeding.
- 4. update_home_member: supply current values for can_control_devices, can_manage_devices, can_manage_members, can_run_routines, home_id, member_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0de9a0a3fc09d434

适用范围：crowdfunding_fundraising_1

When the request requires operations involving: get_campaign_by_id, create_campaign_reward, create_campaign_reward, create_campaign_reward

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_campaign_by_id: supply current values for campaign_id. Inspect the result before proceeding.
- 2. create_campaign_reward: supply current values for campaign_id, min_contribution, quantity_limit, title. Inspect the result before proceeding.
- 3. create_campaign_reward: supply current values for campaign_id, min_contribution, title. Inspect the result before proceeding.
- 4. create_campaign_reward: supply current values for campaign_id, min_contribution, quantity_limit, title. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0e36ae65f1416e8b

适用范围：booking_reservations_29

When the request requires operations involving: list_amenities, search_homes

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_amenities: supply current values for no arguments. Inspect the result before proceeding.
- 2. search_homes: supply current values for amenity_codes, city, country, end_date, guest_count, limit, requires_simultaneous_swap, start_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0f558b720ffd13c7

适用范围：social_dating_5

When the request requires operations involving: list_match_messages, send_message_in_conversation

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_match_messages: supply current values for limit, match_id, offset. Inspect the result before proceeding.
- 2. send_message_in_conversation: supply current values for body, conversation_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0f59eec1a75e4630

适用范围：e_commerce_14

When the request requires operations involving: search_products, create_or_get_wishlist, add_item_to_wishlist

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_products: supply current values for limit, min_average_rating, min_review_count, query, sort_by. Inspect the result before proceeding.
- 2. create_or_get_wishlist: supply current values for name. Inspect the result before proceeding.
- 3. add_item_to_wishlist: supply current values for product_id, wishlist_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_0fc3b09752804257

适用范围：booking_and_reservations_2

When the request requires operations involving: get_primary_restaurant, list_reservations, get_current_shift

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_primary_restaurant: supply current values for no arguments. Inspect the result before proceeding.
- 2. list_reservations: supply current values for min_party_size, restaurant_id, shift_id, status_in. Inspect the result before proceeding.
- 3. get_current_shift: supply current values for current_datetime, restaurant_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_10517cabab913da8

适用范围：subscription_management_6

When the request requires operations involving: list_expiring_titles_by_region_and_genre

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_expiring_titles_by_region_and_genre: supply current values for from_date, genre_name, region_code, to_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_108be6669491dd2a

适用范围：workforce_scheduling_1

When the request requires operations involving: list_roles, bulk_publish_shifts

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_roles: supply current values for organization_id. Inspect the result before proceeding.
- 2. bulk_publish_shifts: supply current values for end_date, location_id, organization_id, require_assigned_employee, required_role_id, start_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_111f784f8504e14f

适用范围：accounting_finance_2

When the request requires operations involving: list_clients, list_tax_codes, create_estimate, create_estimate_line_item, recalculate_estimate_totals

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_clients: supply current values for name. Inspect the result before proceeding.
- 2. list_tax_codes: supply current values for code. Inspect the result before proceeding.
- 3. create_estimate: supply current values for client_id, currency_code, estimate_date, status. Inspect the result before proceeding.
- 4. create_estimate_line_item: supply current values for description, estimate_id, line_number, quantity, tax_code_id, unit_price. Inspect the result before proceeding.
- 5. recalculate_estimate_totals: supply current values for estimate_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_112952e31ea907f8

适用范围：social_network_8

When the request requires operations involving: list_my_applications, list_my_applications_with_opportunity

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_my_applications: supply current values for status_in. Inspect the result before proceeding.
- 2. list_my_applications_with_opportunity: supply current values for status_in. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_11ef9f980ea4b6a2

适用范围：subscription_management_8

When the request requires operations involving: list_entitled_coop_offline_games

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_entitled_coop_offline_games: supply current values for no arguments. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_12481aef35f970cf

适用范围：workflow_automation_2

When the request requires operations involving: list_workflow_versions, list_connectors

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_workflow_versions: supply current values for workflow_id. Inspect the result before proceeding.
- 2. list_connectors: supply current values for name. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_14d77acc4341051a

适用范围：booking_marketplace_4

When the request requires operations involving: list_unpaid_payouts, list_payout_batches

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_unpaid_payouts: supply current values for from_date, to_date. Inspect the result before proceeding.
- 2. list_payout_batches: supply current values for from_date, to_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_14e4df5103dcae0c

适用范围：task_management_11

When the request requires operations involving: list_properties, get_property_dashboard_summary

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_properties: supply current values for name. Inspect the result before proceeding.
- 2. get_property_dashboard_summary: supply current values for property_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_1a71328d26849215

适用范围：marketplace_12

When the request requires operations involving: upsert_course_rating, set_rating_tags, get_course_average_rating

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. upsert_course_rating: supply current values for course_id, short_comment, star_score. Inspect the result before proceeding.
- 2. set_rating_tags: supply current values for rating_id, tag_names. Inspect the result before proceeding.
- 3. get_course_average_rating: supply current values for course_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_1abb2e12710d1d07

适用范围：fleet_management_8

When the request requires operations involving: create_fuel_log

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_fuel_log: supply current values for currency, log_date, odometer_km, price_per_unit, vehicle_id, vendor, volume_liters. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_1bfa53657e90d0b0

适用范围：accounting_finance_2

When the request requires operations involving: list_clients, create_estimate, list_tax_codes, create_estimate_line_item, recalculate_estimate_totals

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_clients: supply current values for name. Inspect the result before proceeding.
- 2. create_estimate: supply current values for client_id, currency_code, estimate_date, status. Inspect the result before proceeding.
- 3. list_tax_codes: supply current values for code. Inspect the result before proceeding.
- 4. create_estimate_line_item: supply current values for description, estimate_id, line_number, quantity, tax_code_id, unit_price. Inspect the result before proceeding.
- 5. recalculate_estimate_totals: supply current values for estimate_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_1c120b5e0eaaf2d9

适用范围：asset_management_7

When the request requires operations involving: get_vehicle_by_asset_id, update_vehicle

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_vehicle_by_asset_id: supply current values for asset_id. Inspect the result before proceeding.
- 2. update_vehicle: supply current values for current_status, department, vehicle_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_1c496d43906b205d

适用范围：booking_reservations_29

When the request requires operations involving: list_house_rules, search_homes

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_house_rules: supply current values for no arguments. Inspect the result before proceeding.
- 2. search_homes: supply current values for amenity_codes, city, country, end_date, guest_count, limit, requires_simultaneous_swap, start_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_1c70dd09eb3ae862

适用范围：e_commerce_9

When the request requires operations involving: list_my_prescriptions, list_family_members, create_family_member, list_my_prescriptions, create_family_prescription_link, create_refill_reminder_setting

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_my_prescriptions: supply current values for is_active. Inspect the result before proceeding.
- 2. list_family_members: supply current values for no arguments. Inspect the result before proceeding.
- 3. create_family_member: supply current values for full_name, is_child, relationship. Inspect the result before proceeding.
- 4. list_my_prescriptions: supply current values for no arguments. Inspect the result before proceeding.
- 5. create_family_prescription_link: supply current values for family_member_id, prescription_id. Inspect the result before proceeding.
- 6. create_refill_reminder_setting: supply current values for days_before, family_member_id, prescription_id, via_sms. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_1de1b500a2f4a39d

适用范围：marketplace_12

When the request requires operations involving: get_or_create_progress_track, list_lesson_blocks, set_lesson_block_completion, recalculate_course_progress

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_or_create_progress_track: supply current values for course_id. Inspect the result before proceeding.
- 2. list_lesson_blocks: supply current values for lesson_id. Inspect the result before proceeding.
- 3. set_lesson_block_completion: supply current values for block_id, completed, completed_at, course_id. Inspect the result before proceeding.
- 4. recalculate_course_progress: supply current values for course_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_1de6b84e0b98ab1d

适用范围：document_contract_management_1

When the request requires operations involving: list_contract_templates, list_counterparties, list_departments, create_contract

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_contract_templates: supply current values for name. Inspect the result before proceeding.
- 2. list_counterparties: supply current values for name. Inspect the result before proceeding.
- 3. list_departments: supply current values for no arguments. Inspect the result before proceeding.
- 4. create_contract: supply current values for auto_renewal_flag, contract_value, counterparty_id, department_id, effective_date, governing_law, region_id, status, template_id, term_length_months. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_20bf6adf9fed521c

适用范围：job_board_3

When the request requires operations involving: get_message_thread_for_application, create_message_in_thread, bulk_update_message_read_status_in_thread

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_message_thread_for_application: supply current values for application_id. Inspect the result before proceeding.
- 2. create_message_in_thread: supply current values for body, sender_role, thread_id. Inspect the result before proceeding.
- 3. bulk_update_message_read_status_in_thread: supply current values for read_status, sender_role, thread_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_2159aa160bb8da58

适用范围：hr_payroll_1

When the request requires operations involving: workflow_apply_bonus_to_pay_statement

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. workflow_apply_bonus_to_pay_statement: supply current values for bonus_amount, bonus_label, employee_id, payroll_run_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_21d85f79e1442092

适用范围：booking_and_scheduling_17

When the request requires operations involving: list_class_types, list_instructors, list_locations, list_rooms, create_class_session

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_class_types: supply current values for name. Inspect the result before proceeding.
- 2. list_instructors: supply current values for full_name. Inspect the result before proceeding.
- 3. list_locations: supply current values for name. Inspect the result before proceeding.
- 4. list_rooms: supply current values for location_id, name. Inspect the result before proceeding.
- 5. create_class_session: supply current values for booking_status, class_type_id, end_time, instructor_id, location_id, room_id, start_time. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_21fe9d9c0eb2b2a6

适用范围：communication_messaging_2

When the request requires operations involving: search_emails, list_labels, add_label_to_email, update_email_star_status

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_emails: supply current values for body_contains, from_address, max_results, sort_by, sort_order, subject_contains. Inspect the result before proceeding.
- 2. list_labels: supply current values for no arguments. Inspect the result before proceeding.
- 3. add_label_to_email: supply current values for email_id, label_id. Inspect the result before proceeding.
- 4. update_email_star_status: supply current values for email_id, is_starred. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_289f006f7c7043c2

适用范围：social_network_8

When the request requires operations involving: list_my_applications_with_opportunity

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_my_applications_with_opportunity: supply current values for status_in. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_28aea5c65abcda84

适用范围：payments_donations_1

When the request requires operations involving: get_campaign_by_name, list_recurring_plans, bulk_pause_recurring_plans_for_donor_campaign

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_campaign_by_name: supply current values for campaign_name. Inspect the result before proceeding.
- 2. list_recurring_plans: supply current values for campaign_id, donor_id, limit. Inspect the result before proceeding.
- 3. bulk_pause_recurring_plans_for_donor_campaign: supply current values for campaign_name, donor_id, note_append. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_292b899e752d6ed6

适用范围：dating_matching_1

When the request requires operations involving: get_match_with_user, bulk_update_date_suggestion_statuses_for_match

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_match_with_user: supply current values for other_user_id. Inspect the result before proceeding.
- 2. bulk_update_date_suggestion_statuses_for_match: supply current values for from_status, match_id, to_status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_294a79489653ac45

适用范围：b2b_saas_4

When the request requires operations involving: get_property_by_address, get_tenant_by_name, get_unit_by_property_and_number, create_lease, create_lease_document

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_property_by_address: supply current values for address_line1, city, postal_code, state. Inspect the result before proceeding.
- 2. get_tenant_by_name: supply current values for name, portfolio_id. Inspect the result before proceeding.
- 3. get_unit_by_property_and_number: supply current values for property_id, unit_number. Inspect the result before proceeding.
- 4. create_lease: supply current values for base_rent_annual, base_rent_currency, billing_schedule, escalation_clauses, lease_end_date, lease_start_date, portfolio_id, property_id, security_deposit, security_deposit_currency, tenant_id, unit_id. Inspect the result before proceeding.
- 5. create_lease_document: supply current values for file_name, lease_id, storage_location. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_295b65a0cbbbef4e

适用范围：competitive_gaming_1

When the request requires operations involving: get_match_by_identifier, get_team_by_id, update_match_result

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_match_by_identifier: supply current values for match_identifier. Inspect the result before proceeding.
- 2. get_team_by_id: supply current values for team_id. Inspect the result before proceeding.
- 3. update_match_result: supply current values for duration_seconds, match_identifier, team1_score, team2_score, winner_team_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_2b5caedfcba34e30

适用范围：fleet_management_8

When the request requires operations involving: create_fuel_log

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_fuel_log: supply current values for log_date, odometer_km, price_per_unit, vehicle_id, vendor, volume_liters. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_2b70c432c56d163d

适用范围：hr_system_1

When the request requires operations involving: list_candidates, list_job_openings, list_applications, create_offer

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_candidates: supply current values for full_name. Inspect the result before proceeding.
- 2. list_job_openings: supply current values for title. Inspect the result before proceeding.
- 3. list_applications: supply current values for candidate_id, job_opening_id. Inspect the result before proceeding.
- 4. create_offer: supply current values for application_id, base_salary, bonus, equity, proposed_start_date, proposed_title, status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_2d38e8bf3ca62b39

适用范围：document_contract_management_1

When the request requires operations involving: list_departments, list_regions, list_counterparties, list_contract_templates, create_contract

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_departments: supply current values for no arguments. Inspect the result before proceeding.
- 2. list_regions: supply current values for no arguments. Inspect the result before proceeding.
- 3. list_counterparties: supply current values for name. Inspect the result before proceeding.
- 4. list_contract_templates: supply current values for name. Inspect the result before proceeding.
- 5. create_contract: supply current values for auto_renewal_flag, contract_value, counterparty_id, department_id, effective_date, governing_law, region_id, status, template_id, term_length_months. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_2e44fe9576205727

适用范围：inventory_management_14

When the request requires operations involving: list_warehouses, list_item_skus, list_bins, list_warehouses, list_item_skus, list_bins, update_inventory_level, get_inventory_available_summary

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_warehouses: supply current values for name. Inspect the result before proceeding.
- 2. list_item_skus: supply current values for sku_code. Inspect the result before proceeding.
- 3. list_bins: supply current values for bin_code, warehouse_id. Inspect the result before proceeding.
- 4. list_warehouses: supply current values for name. Inspect the result before proceeding.
- 5. list_item_skus: supply current values for sku_code. Inspect the result before proceeding.
- 6. list_bins: supply current values for warehouse_id. Inspect the result before proceeding.
- 7. update_inventory_level: supply current values for inventory_level_id, on_hand_qty, reserved_qty. Inspect the result before proceeding.
- 8. get_inventory_available_summary: supply current values for item_sku_id, warehouse_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_308c17d668779752

适用范围：task_management_13

When the request requires operations involving: get_legal_matter_by_title, create_task

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_legal_matter_by_title: supply current values for title. Inspect the result before proceeding.
- 2. create_task: supply current values for assignee, completion_state, due_date, legal_matter_id, title. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_31ad8e97d8d0cc0d

适用范围：collection_management_1

When the request requires operations involving: search_game_titles, bulk_append_notes_for_collection_items

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_game_titles: supply current values for category, min_supports_players, publisher. Inspect the result before proceeding.
- 2. bulk_append_notes_for_collection_items: supply current values for category, min_supported_players, notes_suffix, publisher. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_330f6394a6677679

适用范围：it_service_management_itsm_1

When the request requires operations involving: list_incidents, update_incident_status, create_incident_note

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_incidents: supply current values for incident_number. Inspect the result before proceeding.
- 2. update_incident_status: supply current values for incident_id, new_status. Inspect the result before proceeding.
- 3. create_incident_note: supply current values for expected_update_date, incident_id, note_text, note_type. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_3342dfb3a8353b05

适用范围：banking_6

When the request requires operations involving: list_my_accounts, create_card_for_account, create_card_controls, update_card_status

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_my_accounts: supply current values for no arguments. Inspect the result before proceeding.
- 2. create_card_for_account: supply current values for account_id, card_type, currency, home_country, last4, spending_limit. Inspect the result before proceeding.
- 3. create_card_controls: supply current values for allowed_countries, card_id, international_allowed. Inspect the result before proceeding.
- 4. update_card_status: supply current values for card_id, card_status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_36304fc747154238

适用范围：dating_1

When the request requires operations involving: list_my_matches, list_my_pending_matches

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_my_matches: supply current values for limit, status. Inspect the result before proceeding.
- 2. list_my_pending_matches: supply current values for limit. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_36d2bf8a7b65ff15

适用范围：professional_networking_13

When the request requires operations involving: cancel_match_with_reason

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. cancel_match_with_reason: supply current values for cancel_reason, match_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_380d5b422cc518b8

适用范围：accounting_finance_2

When the request requires operations involving: list_clients, list_tax_codes, create_estimate, create_estimate_line_item

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_clients: supply current values for name. Inspect the result before proceeding.
- 2. list_tax_codes: supply current values for code. Inspect the result before proceeding.
- 3. create_estimate: supply current values for client_id, currency_code, estimate_date, status. Inspect the result before proceeding.
- 4. create_estimate_line_item: supply current values for description, estimate_id, line_number, quantity, tax_code_id, unit_price. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_3857f4e5cb347e1b

适用范围：logistics_and_shipment_tracking_1

When the request requires operations involving: create_webhook_subscription_for_merchant

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_webhook_subscription_for_merchant: supply current values for event_type, merchant_code, status_filter, target_url. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_396e2d108fbe2c1e

适用范围：payments_fintech_2

When the request requires operations involving: list_customers, create_checkout_session

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_customers: supply current values for email. Inspect the result before proceeding.
- 2. create_checkout_session: supply current values for cart_total, currency, customer_id, items_metadata, status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_3a9154aa9801d758

适用范围：billing_payments_4

When the request requires operations involving: list_customers, create_invoice, create_invoice_line_item, recalculate_invoice_totals, finalize_invoice

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_customers: supply current values for no arguments. Inspect the result before proceeding.
- 2. create_invoice: supply current values for currency, customer_id, subscription_id. Inspect the result before proceeding.
- 3. create_invoice_line_item: supply current values for description, invoice_id, quantity, tax_rate, unit_amount. Inspect the result before proceeding.
- 4. recalculate_invoice_totals: supply current values for invoice_id. Inspect the result before proceeding.
- 5. finalize_invoice: supply current values for invoice_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_3f042a70eb325a28

适用范围：social_network_2

When the request requires operations involving: get_guild_by_name, get_guild_channel_by_name, create_channel_post, get_guild_role_by_name, create_post_comment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_guild_by_name: supply current values for guild_name, status. Inspect the result before proceeding.
- 2. get_guild_channel_by_name: supply current values for channel_name, guild_id. Inspect the result before proceeding.
- 3. create_channel_post: supply current values for channel_id, content. Inspect the result before proceeding.
- 4. get_guild_role_by_name: supply current values for guild_id, role_name. Inspect the result before proceeding.
- 5. create_post_comment: supply current values for content, post_id, tagged_role_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_3fab9306d2a6bc23

适用范围：bookmarking_2

When the request requires operations involving: search_creator_profiles, bulk_create_follows

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_creator_profiles: supply current values for category, display_name_contains, language, limit, platform_name. Inspect the result before proceeding.
- 2. bulk_create_follows: supply current values for creator_profile_ids, notify_live, notify_new_video, personal_note. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_431c9477336a141e

适用范围：learning_management_system_3

When the request requires operations involving: list_current_user_valid_certificates

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_current_user_valid_certificates: supply current values for no arguments. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_4381528a2043d698

适用范围：subscription_management_6

When the request requires operations involving: get_my_active_subscription, list_expiring_titles_by_region_and_genre

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_my_active_subscription: supply current values for no arguments. Inspect the result before proceeding.
- 2. list_expiring_titles_by_region_and_genre: supply current values for from_date, genre_name, region_code, to_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_4554ec91e2f7e486

适用范围：social_network_4

When the request requires operations involving: list_communities, create_post

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_communities: supply current values for name. Inspect the result before proceeding.
- 2. create_post: supply current values for body, category, community_id, post_type, title. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_46282d7e265f5f58

适用范围：clinic_management_2

When the request requires operations involving: list_pets, list_vets, create_prescription

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_pets: supply current values for name. Inspect the result before proceeding.
- 2. list_vets: supply current values for full_name. Inspect the result before proceeding.
- 3. create_prescription: supply current values for dosage_instructions, drug_name, pet_id, pharmacy_preference, quantity, refills, vet_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_46eea7c4c9983641

适用范围：booking_marketplace_1

When the request requires operations involving: search_locations, search_listings, create_instant_booking

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_locations: supply current values for city, country. Inspect the result before proceeding.
- 2. search_listings: supply current values for checkin_date, checkout_date, city, country, guests_count, instant_book_only, limit, min_guests, sort_by, sort_order. Inspect the result before proceeding.
- 3. create_instant_booking: supply current values for checkin_date, checkout_date, guests_count, listing_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_47195f30ae517687

适用范围：bookmarking_4

When the request requires operations involving: get_collection_by_title, get_collection_by_title, get_collection_by_title, create_item

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_collection_by_title: supply current values for title. Inspect the result before proceeding.
- 2. get_collection_by_title: supply current values for no arguments. Inspect the result before proceeding.
- 3. get_collection_by_title: supply current values for title. Inspect the result before proceeding.
- 4. create_item: supply current values for collection_id, item_type, price_estimate, rating, short_description, source_url, tags, title. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_4a11d87f87584247

适用范围：learning_management_system_5

When the request requires operations involving: get_submission, get_rubric_criterion_by_name, create_code_review_comment, upsert_rubric_score

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_submission: supply current values for submission_id. Inspect the result before proceeding.
- 2. get_rubric_criterion_by_name: supply current values for assignment_id, name. Inspect the result before proceeding.
- 3. create_code_review_comment: supply current values for comment_text, rubric_criterion_id, submission_id. Inspect the result before proceeding.
- 4. upsert_rubric_score: supply current values for max_score, rubric_criterion_id, score, submission_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_4a16ab8c280d0818

适用范围：e_commerce_digital_marketplace_1

When the request requires operations involving: create_collection, add_product_to_collection

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_collection: supply current values for description, name. Inspect the result before proceeding.
- 2. add_product_to_collection: supply current values for collection_id, product_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_4a2bbb3843b57c59

适用范围：event_management_4

When the request requires operations involving: list_events_for_current_user, list_attendees_for_event, create_checkin

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_events_for_current_user: supply current values for title. Inspect the result before proceeding.
- 2. list_attendees_for_event: supply current values for event_id, limit, offset, ticket_name. Inspect the result before proceeding.
- 3. create_checkin: supply current values for attendee_id, checked_in_at, event_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_4a87173e19d0a198

适用范围：portfolio_management_1

When the request requires operations involving: list_portfolios, list_portfolio_daily_performance

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_portfolios: supply current values for name. Inspect the result before proceeding.
- 2. list_portfolio_daily_performance: supply current values for end_date, portfolio_id, start_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_4f367b734ce428ef

适用范围：insurance_claims_management_2

When the request requires operations involving: get_claim_by_number, create_claim_task

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_claim_by_number: supply current values for claim_number. Inspect the result before proceeding.
- 2. create_claim_task: supply current values for adjuster_id, claim_id, completion_status, due_date, metadata, required_actions. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_4f74b08cd74d4a00

适用范围：social_community_platform_1

When the request requires operations involving: list_institutions, list_my_memberships, list_events, set_event_rsvp, get_event_rsvp_counts

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_institutions: supply current values for name. Inspect the result before proceeding.
- 2. list_my_memberships: supply current values for status. Inspect the result before proceeding.
- 3. list_events: supply current values for organization_id, title. Inspect the result before proceeding.
- 4. set_event_rsvp: supply current values for event_id, response_value. Inspect the result before proceeding.
- 5. get_event_rsvp_counts: supply current values for event_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_576da960166121e5

适用范围：patient_portal_3

When the request requires operations involving: list_primary_care_providers_for_patient, create_appointment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_primary_care_providers_for_patient: supply current values for practice_name. Inspect the result before proceeding.
- 2. create_appointment: supply current values for appointment_datetime, department, duration_minutes, practice_id, provider_id, request_note, specialty, visit_reason. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_59ba3cf91f16c93a

适用范围：billing_and_invoicing_3

When the request requires operations involving: create_payment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_payment: supply current values for amount, bill_id, method, payment_date, reference_number. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_5b08703ed79e8924

适用范围：e_commerce_marketplace_1

When the request requires operations involving: get_vehicle_by_vin, update_vehicle_partial

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_vehicle_by_vin: supply current values for vin. Inspect the result before proceeding.
- 2. update_vehicle_partial: supply current values for metadata, sale_price, status, vin. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_5d6285e4cb7bdb64

适用范围：social_dating_1

When the request requires operations involving: get_recent_incoming_introductions

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_recent_incoming_introductions: supply current values for exclude_expired, hours_lookback. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_5d82da7a85a211ee

适用范围：logistics_management_5

When the request requires operations involving: get_customer_billing_summary_report

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_customer_billing_summary_report: supply current values for end_date, external_account_id, start_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_5fc4a3a9a414f5fc

适用范围：live_streaming_1

When the request requires operations involving: list_live_streams

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_live_streams: supply current values for game_name, limit, sort_by_viewer_count, tag_names. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_615c356351670569

适用范围：billing_and_invoicing_3

When the request requires operations involving: list_bills, create_payment, list_bill_line_items, update_bill

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_bills: supply current values for limit. Inspect the result before proceeding.
- 2. create_payment: supply current values for amount, bill_id, method, payment_date, reference_number. Inspect the result before proceeding.
- 3. list_bill_line_items: supply current values for bill_id. Inspect the result before proceeding.
- 4. update_bill: supply current values for bill_id, payment_status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_62d20d0594e75fa7

适用范围：it_service_management_3

When the request requires operations involving: report_incident_sla_breaches

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. report_incident_sla_breaches: supply current values for days. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_68fc48f5af25b1b2

适用范围：e_commerce_35

When the request requires operations involving: update_my_notification_preferences

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. update_my_notification_preferences: supply current values for invoice_email, invoice_sms, marketing_email, marketing_sms, outbid_email, outbid_sms, win_email, win_sms. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_6a716294779f80aa

适用范围：fundraising_donations_1

When the request requires operations involving: update_event_partial

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. update_event_partial: supply current values for event_date, event_id, fundraising_minimum_amount, location, route_distance_km. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_6a74f48356dc06e8

适用范围：team_scheduling_1

When the request requires operations involving: list_schedules, list_locations_for_organization, list_roles_for_organization, bulk_create_time_slots

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_schedules: supply current values for name. Inspect the result before proceeding.
- 2. list_locations_for_organization: supply current values for name, organization_id. Inspect the result before proceeding.
- 3. list_roles_for_organization: supply current values for name, organization_id. Inspect the result before proceeding.
- 4. bulk_create_time_slots: supply current values for time_slots. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_6cfdff3103dd9492

适用范围：ats_applicant_tracking_system_2

When the request requires operations involving: create_application

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_application: supply current values for candidate_id, current_stage, id, job_requisition_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_6f38282da3cb1625

适用范围：social_platform_1

When the request requires operations involving: list_room_sessions_for_owner, list_owned_rooms, add_session_playback_state, create_chat_message

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_room_sessions_for_owner: supply current values for title. Inspect the result before proceeding.
- 2. list_owned_rooms: supply current values for title. Inspect the result before proceeding.
- 3. add_session_playback_state: supply current values for session_id, status, timestamp_seconds. Inspect the result before proceeding.
- 4. create_chat_message: supply current values for content, moderation_flag, playback_timestamp_seconds, session_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_6f3ff1305eacd2b0

适用范围：iot_smart_device_management_1

When the request requires operations involving: list_sites, list_maintenance_tasks, bulk_reschedule_overdue_tasks

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_sites: supply current values for name. Inspect the result before proceeding.
- 2. list_maintenance_tasks: supply current values for priority_min, site_id. Inspect the result before proceeding.
- 3. bulk_reschedule_overdue_tasks: supply current values for current_timestamp, extend_by_days, min_priority, note_text, site_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_70abc12c2d73715d

适用范围：asset_management_6

When the request requires operations involving: list_assets, list_shops, list_service_reminders, create_work_order, create_work_order_task

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_assets: supply current values for vin. Inspect the result before proceeding.
- 2. list_shops: supply current values for type. Inspect the result before proceeding.
- 3. list_service_reminders: supply current values for asset_id. Inspect the result before proceeding.
- 4. create_work_order: supply current values for asset_id, estimated_total_cost, opened_at, service_reminder_id, shop_id, status, title. Inspect the result before proceeding.
- 5. create_work_order_task: supply current values for description, estimated_cost, position, work_order_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_70b861c5a52aeb43

适用范围：dating_1

When the request requires operations involving: list_my_matches

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_my_matches: supply current values for status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_71e1621b96d6f7e1

适用范围：marketplace_12

When the request requires operations involving: upsert_course_rating, create_or_get_rating_tag, get_course_average_rating

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. upsert_course_rating: supply current values for course_id, short_comment, star_score. Inspect the result before proceeding.
- 2. create_or_get_rating_tag: supply current values for name. Inspect the result before proceeding.
- 3. get_course_average_rating: supply current values for course_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_722397a643cd1864

适用范围：b2b_saas_4

When the request requires operations involving: get_property_by_address, get_unit_by_property_and_number, list_property_units, create_lease, create_lease_document

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_property_by_address: supply current values for address_line1, city, postal_code, state. Inspect the result before proceeding.
- 2. get_unit_by_property_and_number: supply current values for property_id, unit_number. Inspect the result before proceeding.
- 3. list_property_units: supply current values for property_id. Inspect the result before proceeding.
- 4. create_lease: supply current values for base_rent_annual, base_rent_currency, billing_schedule, escalation_clauses, lease_end_date, lease_start_date, portfolio_id, property_id, security_deposit, security_deposit_currency, tenant_id, unit_id. Inspect the result before proceeding.
- 5. create_lease_document: supply current values for file_name, lease_id, storage_location. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_74b09c275c7d7758

适用范围：booking_marketplace_4

When the request requires operations involving: list_unpaid_payouts, list_unpaid_payouts

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_unpaid_payouts: supply current values for from_date, to_date. Inspect the result before proceeding.
- 2. list_unpaid_payouts: supply current values for no arguments. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_75b0c3e4fb27f279

适用范围：banking_4

When the request requires operations involving: create_loan_calculator_estimate

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_loan_calculator_estimate: supply current values for apr, loan_type, name, principal_amount, term_months. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_75b1d9dde2c1d990

适用范围：inventory_management_1

When the request requires operations involving: list_locations, get_reorder_rule_by_location_and_sku, patch_reorder_rule_by_id, evaluate_reorder_for_location_and_sku

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_locations: supply current values for location_name. Inspect the result before proceeding.
- 2. get_reorder_rule_by_location_and_sku: supply current values for internal_sku, location_name. Inspect the result before proceeding.
- 3. patch_reorder_rule_by_id: supply current values for max_quantity, min_quantity, preferred_supplier_name, reorder_rule_id. Inspect the result before proceeding.
- 4. evaluate_reorder_for_location_and_sku: supply current values for internal_sku, location_name. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_761e0a992ecbdb76

适用范围：loyalty_and_rewards_2

When the request requires operations involving: list_brands_for_current_user, create_earn_rule

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_brands_for_current_user: supply current values for no arguments. Inspect the result before proceeding.
- 2. create_earn_rule: supply current values for currency_code, description, is_active, min_order_amount, name, points_per_currency_unit, program_id, rule_type, valid_from, valid_to. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_763e949bbf5724a0

适用范围：patient_portal_3

When the request requires operations involving: list_primary_care_providers_for_patient, get_available_appointment_slots, create_appointment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_primary_care_providers_for_patient: supply current values for practice_name. Inspect the result before proceeding.
- 2. get_available_appointment_slots: supply current values for end_datetime, practice_id, preferred_time_of_day, provider_id, required_duration_minutes, start_datetime. Inspect the result before proceeding.
- 3. create_appointment: supply current values for appointment_datetime, duration_minutes, practice_id, provider_id, request_note, specialty, visit_reason. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_772440d247ea10ab

适用范围：booking_appointments_3

When the request requires operations involving: list_clinics, list_veterinarians, add_veterinarian_specialty, set_weekday_working_hours

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_clinics: supply current values for name. Inspect the result before proceeding.
- 2. list_veterinarians: supply current values for clinic_id. Inspect the result before proceeding.
- 3. add_veterinarian_specialty: supply current values for specialty, veterinarian_id. Inspect the result before proceeding.
- 4. set_weekday_working_hours: supply current values for end_time, start_time, timezone, veterinarian_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_777357edad2689b3

适用范围：booking_marketplace_1

When the request requires operations involving: search_listings, create_instant_booking

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_listings: supply current values for checkin_date, checkout_date, city, country, free_cancellation_min_days, guests_count, instant_book_only, limit, sort_by, sort_order. Inspect the result before proceeding.
- 2. create_instant_booking: supply current values for checkin_date, checkout_date, guests_count, listing_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_788238f2778753a9

适用范围：ordering_and_payments_1

When the request requires operations involving: list_venues, list_orders_for_venue, get_pickup_ticket_with_order_state

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_venues: supply current values for name. Inspect the result before proceeding.
- 2. list_orders_for_venue: supply current values for venue_id. Inspect the result before proceeding.
- 3. get_pickup_ticket_with_order_state: supply current values for order_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_7b881fc593715d10

适用范围：social_dating_5

When the request requires operations involving: list_match_messages, send_match_message

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_match_messages: supply current values for limit, match_id, offset. Inspect the result before proceeding.
- 2. send_match_message: supply current values for body, match_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_7befb15c685abfa3

适用范围：iot_smart_home_control_5

When the request requires operations involving: list_owned_homes, get_room_by_name, list_home_members, update_home_member

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_owned_homes: supply current values for no arguments. Inspect the result before proceeding.
- 2. get_room_by_name: supply current values for home_id, room_name. Inspect the result before proceeding.
- 3. list_home_members: supply current values for home_id. Inspect the result before proceeding.
- 4. update_home_member: supply current values for can_control_devices, can_manage_devices, can_manage_members, can_run_routines, home_id, member_id, role. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_7c034d5cbbb9a99b

适用范围：subscription_management_8

When the request requires operations involving: list_user_entitlements, create_play_session

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_user_entitlements: supply current values for is_active. Inspect the result before proceeding.
- 2. create_play_session: supply current values for approximate_progress_percent, device_type, end_time, entitlement_id, game_id, start_time. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_7cc9dbd7872c1df4

适用范围：booking_platform_5

When the request requires operations involving: create_pet_for_my_owner_profile

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_pet_for_my_owner_profile: supply current values for age_years, behavior_flags, feeding_schedule, health_notes, name, species. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_7cf6881abab2f51c

适用范围：hr_system_1

When the request requires operations involving: list_candidates, list_job_openings, create_offer

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_candidates: supply current values for full_name. Inspect the result before proceeding.
- 2. list_job_openings: supply current values for title. Inspect the result before proceeding.
- 3. create_offer: supply current values for application_id, base_salary, bonus, equity, proposed_start_date, proposed_title, status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_804425e0e1a28315

适用范围：payments_donations_1

When the request requires operations involving: bulk_pause_recurring_plans_for_donor_campaign

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. bulk_pause_recurring_plans_for_donor_campaign: supply current values for campaign_name, donor_id, note_append. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_81fa7fdfe88623b0

适用范围：e_commerce_37

When the request requires operations involving: list_merchant_stores, create_product, create_product_variant

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_merchant_stores: supply current values for no arguments. Inspect the result before proceeding.
- 2. create_product: supply current values for base_price, fulfillment_type, merchant_store_id, pet_type, short_description, tag_names, title. Inspect the result before proceeding.
- 3. create_product_variant: supply current values for barcode_metadata, price, product_id, reorder_threshold, size_label, sku, stock_on_hand, weight_unit, weight_value. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_874a5922004baf94

适用范围：hr_management_1

When the request requires operations involving: list_supervisory_organizations, create_job_requisition, create_job_requisition_posting

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_supervisory_organizations: supply current values for location, name. Inspect the result before proceeding.
- 2. create_job_requisition: supply current values for currency, location, requisition_number, salary_max, salary_min, supervisory_org_id, target_hire_date, title. Inspect the result before proceeding.
- 3. create_job_requisition_posting: supply current values for job_requisition_id, site_type. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_89fed1e39036024b

适用范围：donation_fundraising_management_1

When the request requires operations involving: update_current_user_profile

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. update_current_user_profile: supply current values for bio, default_currency, display_name. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_8d8aa095c0f65f0e

适用范围：bookmarking_2

When the request requires operations involving: search_creator_profiles, create_follow

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_creator_profiles: supply current values for category, display_name_contains, language, limit, offset, platform_name. Inspect the result before proceeding.
- 2. create_follow: supply current values for creator_profile_id, notify_live, notify_new_video, personal_note. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_8fb1d6d1e708fb45

适用范围：e_commerce_14

When the request requires operations involving: search_products, create_or_get_wishlist, add_item_to_wishlist

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_products: supply current values for category_name, limit, min_average_rating, min_review_count, query, sort_by. Inspect the result before proceeding.
- 2. create_or_get_wishlist: supply current values for name. Inspect the result before proceeding.
- 3. add_item_to_wishlist: supply current values for product_id, wishlist_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_9021c104c9c352cd

适用范围：banking_6

When the request requires operations involving: get_account_by_last4_for_current_user, create_card_for_account, update_card_status, create_card_controls

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_account_by_last4_for_current_user: supply current values for account_last4. Inspect the result before proceeding.
- 2. create_card_for_account: supply current values for account_id, card_type, currency, last4, spending_limit. Inspect the result before proceeding.
- 3. update_card_status: supply current values for card_id, card_status. Inspect the result before proceeding.
- 4. create_card_controls: supply current values for allowed_countries, card_id, international_allowed. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_9217a8b40f57c409

适用范围：iot_smart_home_industrial_1

When the request requires operations involving: list_maintenance_tickets, update_maintenance_ticket

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_maintenance_tickets: supply current values for created_after, created_before, limit, status. Inspect the result before proceeding.
- 2. update_maintenance_ticket: supply current values for status, tdid, ticket_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_92d89fafc67f6c38

适用范围：booking_and_reservations_2

When the request requires operations involving: get_primary_restaurant, list_reservations, list_reservations

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_primary_restaurant: supply current values for no arguments. Inspect the result before proceeding.
- 2. list_reservations: supply current values for end_datetime, min_party_size, restaurant_id, start_datetime, status_in. Inspect the result before proceeding.
- 3. list_reservations: supply current values for end_datetime, min_party_size, restaurant_id, shift_id, start_datetime, status_in. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_94b1f6afb6677521

适用范围：job_board_1

When the request requires operations involving: get_current_student_profile, list_majors, add_major_to_current_student, patch_current_student_profile

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_current_student_profile: supply current values for no arguments. Inspect the result before proceeding.
- 2. list_majors: supply current values for name. Inspect the result before proceeding.
- 3. add_major_to_current_student: supply current values for major_id. Inspect the result before proceeding.
- 4. patch_current_student_profile: supply current values for preferred_job_type_coop, preferred_job_type_internship. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_950cd60c840c2e0a

适用范围：live_streaming_1

When the request requires operations involving: list_tags, list_live_streams

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_tags: supply current values for limit, name. Inspect the result before proceeding.
- 2. list_live_streams: supply current values for game_name, language_code, limit, offset, sort_by_viewer_count, tag_ids. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_953f370da7f9246f

适用范围：fleet_management_7

When the request requires operations involving: get_primary_fleet, get_vehicle_by_plate, update_vehicle_odometer_status

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_primary_fleet: supply current values for no arguments. Inspect the result before proceeding.
- 2. get_vehicle_by_plate: supply current values for plate_number. Inspect the result before proceeding.
- 3. update_vehicle_odometer_status: supply current values for odometer_km, status, vehicle_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_96c22e38bd666be1

适用范围：e_commerce_14

When the request requires operations involving: search_products, create_or_get_wishlist, add_item_to_wishlist

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_products: supply current values for category_name, get_product_review_stats, limit, query, sort_by. Inspect the result before proceeding.
- 2. create_or_get_wishlist: supply current values for name. Inspect the result before proceeding.
- 3. add_item_to_wishlist: supply current values for product_id, wishlist_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_979bf96d9167f6de

适用范围：application_management_3

When the request requires operations involving: list_program_applications_with_stats, bulk_update_application_statuses_for_program

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_program_applications_with_stats: supply current values for limit, max_average_score, program_id, status. Inspect the result before proceeding.
- 2. bulk_update_application_statuses_for_program: supply current values for current_status, decision_note, max_average_score, min_average_score, new_status, program_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_9bdeced94a1a9061

适用范围：bookmark_manager_1

When the request requires operations involving: create_collection, create_wishlist_item

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_collection: supply current values for collection_type, description, name, visibility. Inspect the result before proceeding.
- 2. create_wishlist_item: supply current values for collection_id, is_completed, target_date, url. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_a0c9e79cf10503d0

适用范围：dating_1

When the request requires operations involving: list_my_matches

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_my_matches: supply current values for created_since_hours, limit, offset, status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_a275f97406d553bf

适用范围：e_commerce_14

When the request requires operations involving: search_products, search_products, create_or_get_wishlist, add_item_to_wishlist

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_products: supply current values for category_name, limit, min_average_rating, min_review_count, query, sort_by. Inspect the result before proceeding.
- 2. search_products: supply current values for limit, min_average_rating, min_review_count, query, sort_by. Inspect the result before proceeding.
- 3. create_or_get_wishlist: supply current values for name. Inspect the result before proceeding.
- 4. add_item_to_wishlist: supply current values for product_id, wishlist_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_a5873f5647d65659

适用范围：insurance_policy_administration_1

When the request requires operations involving: get_policy_by_number, add_policy_coverage

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_policy_by_number: supply current values for policy_number. Inspect the result before proceeding.
- 2. add_policy_coverage: supply current values for coverage_subtype, deductible, limit_amount, policy_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_a720867053d60bfc

适用范围：billing_and_invoicing_3

When the request requires operations involving: get_bill_by_id, create_payment, update_bill

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_bill_by_id: supply current values for bill_id. Inspect the result before proceeding.
- 2. create_payment: supply current values for amount, bill_id, method, payment_date, reference_number. Inspect the result before proceeding.
- 3. update_bill: supply current values for bill_id, payment_status, total_amount. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_a7420fc3a74ffee5

适用范围：learning_management_system_3

When the request requires operations involving: list_current_user_completed_enrollments, list_current_user_valid_certificates

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_current_user_completed_enrollments: supply current values for no arguments. Inspect the result before proceeding.
- 2. list_current_user_valid_certificates: supply current values for no arguments. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_a77b2965742bd264

适用范围：insurance_policy_administration_1

When the request requires operations involving: list_business_customers, create_quote

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_business_customers: supply current values for name. Inspect the result before proceeding.
- 2. create_quote: supply current values for business_customer_id, coverage_options, coverage_type, rating_parameters, status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_a9c63b22ab7efc7b

适用范围：live_streaming_1

When the request requires operations involving: list_tags, list_tags, list_live_streams

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_tags: supply current values for name. Inspect the result before proceeding.
- 2. list_tags: supply current values for limit, search. Inspect the result before proceeding.
- 3. list_live_streams: supply current values for game_name, language_code, limit, sort_by_viewer_count, tag_ids. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_a9f3433d26e437ba

适用范围：fleet_management_7

When the request requires operations involving: get_vehicle_by_plate, update_vehicle_odometer_status

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_vehicle_by_plate: supply current values for plate_number. Inspect the result before proceeding.
- 2. update_vehicle_odometer_status: supply current values for odometer_km, status, vehicle_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_aa8253e679afcd2f

适用范围：property_management_1

When the request requires operations involving: list_late_rent_tenants_by_portfolio, list_notice_templates, create_notice, create_tenant_delivery_preference

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_late_rent_tenants_by_portfolio: supply current values for as_of_date, days_past_due, days_past_due_note, portfolio_name. Inspect the result before proceeding.
- 2. list_notice_templates: supply current values for name. Inspect the result before proceeding.
- 3. create_notice: supply current values for as_of_date, notice_type, template_id, tenant_id. Inspect the result before proceeding.
- 4. create_tenant_delivery_preference: supply current values for channel, enabled, tenant_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_acc8f8341d899159

适用范围：hr_management_1

When the request requires operations involving: list_supervisory_organizations, create_job_requisition, create_job_requisition_posting

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_supervisory_organizations: supply current values for location, name. Inspect the result before proceeding.
- 2. create_job_requisition: supply current values for currency, location, requisition_number, salary_max, salary_min, status, supervisory_org_id, target_hire_date, title. Inspect the result before proceeding.
- 3. create_job_requisition_posting: supply current values for job_requisition_id, site_type. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_aea27d572840049c

适用范围：billing_and_invoicing_3

When the request requires operations involving: get_bill_by_id, create_payment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_bill_by_id: supply current values for bill_id. Inspect the result before proceeding.
- 2. create_payment: supply current values for amount, bill_id, method, payment_date, reference_number. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_af99fffa01684c6a

适用范围：dating_3

When the request requires operations involving: create_my_user_preferences

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_my_user_preferences: supply current values for age_max, age_min, max_distance_km, require_non_smoker, require_ok_pets, want_children_future. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_b157a52292420b2a

适用范围：e_commerce_marketplace_21

When the request requires operations involving: find_cheapest_active_listings_by_catalog_filters

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. find_cheapest_active_listings_by_catalog_filters: supply current values for category_name, limit, rarity_id, supported_app_tag. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_b1b50a23e7150c62

适用范围：document_management_2

When the request requires operations involving: generate_and_send_envelope_from_template

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. generate_and_send_envelope_from_template: supply current values for role_assignments, subject, template_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_b91da0ccfd4a762e

适用范围：booking_marketplace_3

When the request requires operations involving: upsert_renter_profile, set_renter_lifestyle_tags

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. upsert_renter_profile: supply current values for budget_max, budget_min, target_move_in_date. Inspect the result before proceeding.
- 2. set_renter_lifestyle_tags: supply current values for tag_names. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_bc8bda70208be52b

适用范围：survey_and_forms_2

When the request requires operations involving: create_form_template, create_form_template_section, create_form_template_question

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_form_template: supply current values for name, workspace_id. Inspect the result before proceeding.
- 2. create_form_template_section: supply current values for display_order, form_template_id, title. Inspect the result before proceeding.
- 3. create_form_template_question: supply current values for display_order, form_template_id, is_required, label, question_type, section_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_bd43c31958de9a9d

适用范围：bookmarking_2

When the request requires operations involving: search_creator_profiles, bulk_create_follows

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_creator_profiles: supply current values for category, display_name_contains, language, limit, offset, platform_name. Inspect the result before proceeding.
- 2. bulk_create_follows: supply current values for creator_profile_ids, notify_live, notify_new_video, personal_note. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_bdb44d6e6b208562

适用范围：ordering_and_payments_1

When the request requires operations involving: list_venues, get_daily_summary_report

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_venues: supply current values for name. Inspect the result before proceeding.
- 2. get_daily_summary_report: supply current values for date, venue_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_bf22403cdcfbf92d

适用范围：appointment_booking_2

When the request requires operations involving: get_pet_by_name_for_owner, get_clinic_by_name, get_vet_by_name_and_clinic

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_pet_by_name_for_owner: supply current values for pet_name. Inspect the result before proceeding.
- 2. get_clinic_by_name: supply current values for clinic_name. Inspect the result before proceeding.
- 3. get_vet_by_name_and_clinic: supply current values for full_name. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_c1ae9b02f0e187f1

适用范围：bookmarking_4

When the request requires operations involving: get_collection_by_title, create_item

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_collection_by_title: supply current values for title. Inspect the result before proceeding.
- 2. create_item: supply current values for collection_id, item_type, price_estimate, rating, short_description, source_url, tags, title. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_c54ef5e14f2142e0

适用范围：booking_and_reservations_11

When the request requires operations involving: list_locations, create_client, list_staff, upsert_client_preference

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_locations: supply current values for name. Inspect the result before proceeding.
- 2. create_client: supply current values for email, full_name, notes, phone, preferred_location_id. Inspect the result before proceeding.
- 3. list_staff: supply current values for full_name. Inspect the result before proceeding.
- 4. upsert_client_preference: supply current values for client_id, notes, preferred_staff_id, service_category_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_cb7bc3eca299841f

适用范围：payments_fintech_2

When the request requires operations involving: get_current_user, list_customers, create_checkout_session

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_current_user: supply current values for no arguments. Inspect the result before proceeding.
- 2. list_customers: supply current values for email. Inspect the result before proceeding.
- 3. create_checkout_session: supply current values for cart_total, currency, customer_id, items_metadata, status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_cd86cbb5233ddfab

适用范围：e_commerce_21

When the request requires operations involving: update_payment, get_order_shipment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. update_payment: supply current values for metadata_json, payment_id, status. Inspect the result before proceeding.
- 2. get_order_shipment: supply current values for order_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d0aeb38af3167ad4

适用范围：subscription_management_8

When the request requires operations involving: get_user_entitlement_for_game, create_play_session

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_user_entitlement_for_game: supply current values for game_id. Inspect the result before proceeding.
- 2. create_play_session: supply current values for approximate_progress_percent, device_type, end_time, entitlement_id, game_id, start_time. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d0af7f81653bfe6b

适用范围：social_curation_1

When the request requires operations involving: get_my_board_by_title, list_categories, create_entry_on_board, create_entry_link

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_my_board_by_title: supply current values for board_title. Inspect the result before proceeding.
- 2. list_categories: supply current values for name_contains. Inspect the result before proceeding.
- 3. create_entry_on_board: supply current values for board_id, category_id, rating, short_reason_text, title. Inspect the result before proceeding.
- 4. create_entry_link: supply current values for entry_id, link_type, url. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d11beca5a7381452

适用范围：social_networking_5

When the request requires operations involving: search_companies, follow_company, create_job_alert

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. search_companies: supply current values for name_query. Inspect the result before proceeding.
- 2. follow_company: supply current values for company_id. Inspect the result before proceeding.
- 3. create_job_alert: supply current values for company_id, is_active, title_keyword. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d1cd3adc188f264f

适用范围：e_commerce_21

When the request requires operations involving: get_payment, update_payment, get_order_shipment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_payment: supply current values for payment_id. Inspect the result before proceeding.
- 2. update_payment: supply current values for metadata_json, payment_id, status. Inspect the result before proceeding.
- 3. get_order_shipment: supply current values for order_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d28f0634ffeeb100

适用范围：form_builder_5

When the request requires operations involving: list_response_submissions, list_submissions_with_answers

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_response_submissions: supply current values for channel, form_id, status, submitted_from, submitted_to. Inspect the result before proceeding.
- 2. list_submissions_with_answers: supply current values for channel, form_id, status, submitted_from, submitted_to. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d45bf657c6ea1c51

适用范围：workflow_task_management_1

When the request requires operations involving: list_application_cycles, get_cycle_submissions_with_scores_by_org

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_application_cycles: supply current values for name. Inspect the result before proceeding.
- 2. get_cycle_submissions_with_scores_by_org: supply current values for cycle_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d4e5bda7aedd4d80

适用范围：workforce_scheduling_1

When the request requires operations involving: get_organization, list_roles, bulk_publish_shifts

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_organization: supply current values for organization_id. Inspect the result before proceeding.
- 2. list_roles: supply current values for organization_id. Inspect the result before proceeding.
- 3. bulk_publish_shifts: supply current values for end_date, location_id, organization_id, require_assigned_employee, required_role_id, start_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d68eb886c71123f2

适用范围：inventory_management_1

When the request requires operations involving: get_reorder_rule_by_location_and_sku, patch_reorder_rule_by_id, evaluate_reorder_for_location_and_sku

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_reorder_rule_by_location_and_sku: supply current values for internal_sku, location_name. Inspect the result before proceeding.
- 2. patch_reorder_rule_by_id: supply current values for max_quantity, min_quantity, preferred_supplier_name, reorder_rule_id. Inspect the result before proceeding.
- 3. evaluate_reorder_for_location_and_sku: supply current values for internal_sku, location_name. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d74c531fbc7d40d0

适用范围：b2b_saas_4

When the request requires operations involving: list_portfolios, list_tenants, get_property_by_address, get_unit_by_property_and_number, get_tenant_by_name, create_lease, create_lease_document

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_portfolios: supply current values for name. Inspect the result before proceeding.
- 2. list_tenants: supply current values for name. Inspect the result before proceeding.
- 3. get_property_by_address: supply current values for address_line1, city, postal_code, state. Inspect the result before proceeding.
- 4. get_unit_by_property_and_number: supply current values for property_id, unit_number. Inspect the result before proceeding.
- 5. get_tenant_by_name: supply current values for name, portfolio_id. Inspect the result before proceeding.
- 6. create_lease: supply current values for base_rent_annual, base_rent_currency, billing_schedule, escalation_clauses, lease_end_date, lease_start_date, portfolio_id, property_id, security_deposit, security_deposit_currency, tenant_id, unit_id. Inspect the result before proceeding.
- 7. create_lease_document: supply current values for file_name, lease_id, storage_location. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d8a452cc4b44a868

适用范围：education_lms_1

When the request requires operations involving: get_course, create_pricing_plan, create_discount_rule

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_course: supply current values for course_id. Inspect the result before proceeding.
- 2. create_pricing_plan: supply current values for access_duration_days, course_id, currency, name, price. Inspect the result before proceeding.
- 3. create_discount_rule: supply current values for coupon_code, discount_percent, pricing_plan_id, valid_until. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_d9f0ac7618bbc721

适用范围：social_curation_1

When the request requires operations involving: get_my_board_by_title, create_or_get_category, create_entry_on_board, search_entries, create_entry_link

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_my_board_by_title: supply current values for board_title. Inspect the result before proceeding.
- 2. create_or_get_category: supply current values for name. Inspect the result before proceeding.
- 3. create_entry_on_board: supply current values for board_id, category_id, rating, short_reason_text, title. Inspect the result before proceeding.
- 4. search_entries: supply current values for category_name, limit, min_rating, offset, sort_by. Inspect the result before proceeding.
- 5. create_entry_link: supply current values for entry_id, link_type, url. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_da1551a0d44299e1

适用范围：ats_applicant_tracking_system_2

When the request requires operations involving: get_current_user, create_application

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_current_user: supply current values for no arguments. Inspect the result before proceeding.
- 2. create_application: supply current values for candidate_id, current_stage, id, job_requisition_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_df63e662a9082b88

适用范围：job_board_2

When the request requires operations involving: filter_applications_for_employer_job_post

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. filter_applications_for_employer_job_post: supply current values for job_post_id, max_salary_expectation_amount, required_candidate_skills_all, salary_expectation_currency, status_in. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_e2afd5961ed72530

适用范围：e_commerce_21

When the request requires operations involving: update_payment, get_payment, get_order, get_order_shipment

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. update_payment: supply current values for metadata_json, payment_id, status. Inspect the result before proceeding.
- 2. get_payment: supply current values for payment_id. Inspect the result before proceeding.
- 3. get_order: supply current values for order_id. Inspect the result before proceeding.
- 4. get_order_shipment: supply current values for order_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_e32fbf8135d6da25

适用范围：education_lms_1

When the request requires operations involving: list_course_pricing_plans, create_pricing_plan, create_discount_rule

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_course_pricing_plans: supply current values for course_id. Inspect the result before proceeding.
- 2. create_pricing_plan: supply current values for access_duration_days, course_id, currency, name, price. Inspect the result before proceeding.
- 3. create_discount_rule: supply current values for coupon_code, discount_percent, pricing_plan_id, valid_until. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_e5b78690f25c0206

适用范围：education_lms_1

When the request requires operations involving: create_pricing_plan, create_discount_rule

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. create_pricing_plan: supply current values for access_duration_days, course_id, currency, name, price. Inspect the result before proceeding.
- 2. create_discount_rule: supply current values for coupon_code, discount_percent, pricing_plan_id, valid_until. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_e9e8c51aa645a913

适用范围：social_platform_1

When the request requires operations involving: list_room_sessions_for_owner, list_owned_rooms, list_room_sessions_for_owner, list_room_sessions_for_owner, add_session_playback_state, create_chat_message

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_room_sessions_for_owner: supply current values for title. Inspect the result before proceeding.
- 2. list_owned_rooms: supply current values for title. Inspect the result before proceeding.
- 3. list_room_sessions_for_owner: supply current values for room_id. Inspect the result before proceeding.
- 4. list_room_sessions_for_owner: supply current values for no arguments. Inspect the result before proceeding.
- 5. add_session_playback_state: supply current values for session_id, status, timestamp_seconds. Inspect the result before proceeding.
- 6. create_chat_message: supply current values for content, moderation_flag, playback_timestamp_seconds, session_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_ed83f573aa555403

适用范围：e_commerce_ordering_4

When the request requires operations involving: get_my_customer_profile, list_customer_addresses, list_my_restaurants, get_my_customer_profile, list_customer_addresses

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_my_customer_profile: supply current values for no arguments. Inspect the result before proceeding.
- 2. list_customer_addresses: supply current values for customer_id. Inspect the result before proceeding.
- 3. list_my_restaurants: supply current values for no arguments. Inspect the result before proceeding.
- 4. get_my_customer_profile: supply current values for no arguments. Inspect the result before proceeding.
- 5. list_customer_addresses: supply current values for no arguments. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f086826bd985747a

适用范围：booking_reservations_12

When the request requires operations involving: get_space_available_time_slots

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_space_available_time_slots: supply current values for date, space_id, user_role. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f171b13bdcf8b9f7

适用范围：dating_matching_1

When the request requires operations involving: get_match_with_user, get_latest_incoming_date_suggestion, bulk_update_date_suggestion_statuses_for_match

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_match_with_user: supply current values for other_user_id. Inspect the result before proceeding.
- 2. get_latest_incoming_date_suggestion: supply current values for match_id. Inspect the result before proceeding.
- 3. bulk_update_date_suggestion_statuses_for_match: supply current values for from_status, match_id, to_status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f26d4c0ce3dfdaed

适用范围：customer_support_3

When the request requires operations involving: get_kb_category_by_name, create_kb_article, add_tags_to_kb_article

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_kb_category_by_name: supply current values for name. Inspect the result before proceeding.
- 2. create_kb_article: supply current values for body, category_id, summary, title, visibility. Inspect the result before proceeding.
- 3. add_tags_to_kb_article: supply current values for article_id, tag_names. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f2b48ff9dd02d36c

适用范围：job_board_1

When the request requires operations involving: list_majors, add_major_to_current_student, patch_current_student_profile

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_majors: supply current values for name. Inspect the result before proceeding.
- 2. add_major_to_current_student: supply current values for major_id. Inspect the result before proceeding.
- 3. patch_current_student_profile: supply current values for preferred_job_type_coop, preferred_job_type_internship. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f497a90b47a44e31

适用范围：fundraising_donations_1

When the request requires operations involving: update_event_partial

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. update_event_partial: supply current values for event_date, event_id, fundraising_minimum_amount, fundraising_minimum_currency, location, registered_fee_amount, registration_fee_currency, route_distance_km. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f60dbe4c7055848b

适用范围：e_commerce_8

When the request requires operations involving: get_recent_undelivered_order, get_order_shipments

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_recent_undelivered_order: supply current values for no arguments. Inspect the result before proceeding.
- 2. get_order_shipments: supply current values for order_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：4 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f894771800fdb3b7

适用范围：education_2

When the request requires operations involving: list_sections, get_section_combined_grade_attendance_summary

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_sections: supply current values for section_code. Inspect the result before proceeding.
- 2. get_section_combined_grade_attendance_summary: supply current values for from_date, section_id, to_date. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f90318468ca71420

适用范围：dating_1

When the request requires operations involving: list_my_matches

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_my_matches: supply current values for limit, status. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f981185b037806ac

适用范围：crowdfunding_fundraising_1

When the request requires operations involving: get_campaign_by_id, create_campaign_reward, create_campaign_reward, create_campaign_reward

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_campaign_by_id: supply current values for campaign_id. Inspect the result before proceeding.
- 2. create_campaign_reward: supply current values for campaign_id, currency, min_contribution, quantity_limit, title. Inspect the result before proceeding.
- 3. create_campaign_reward: supply current values for campaign_id, currency, min_contribution, title. Inspect the result before proceeding.
- 4. create_campaign_reward: supply current values for campaign_id, currency, min_contribution, quantity_limit, title. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：3 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_f9dcaeabde5bc324

适用范围：competitive_gaming_1

When the request requires operations involving: get_match_by_identifier, update_match_result

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. get_match_by_identifier: supply current values for match_identifier. Inspect the result before proceeding.
- 2. update_match_result: supply current values for duration_seconds, match_identifier, team1_score, team2_score, winner_team_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：2 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_fad7a4501842669b

适用范围：live_streaming_1

When the request requires operations involving: list_live_streams

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_live_streams: supply current values for game_name, language_code, limit, sort_by_viewer_count, tag_names. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_fb187958434536de

适用范围：b2b_payments_1

When the request requires operations involving: list_vendors, create_payment, create_bill_payment_link

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_vendors: supply current values for search_name. Inspect the result before proceeding.
- 2. create_payment: supply current values for amount, currency, destination_account_id, id, scheduled_date, settlement_date, settlement_status, source_account_id. Inspect the result before proceeding.
- 3. create_bill_payment_link: supply current values for applied_amount, bill_id, payment_id. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。

## workflow_ffa43c74194fb28a

适用范围：booking_and_scheduling_17

When the request requires operations involving: list_class_types, list_locations, list_rooms, list_instructors, create_class_session

- Read the schemas of the relevant tools below; adapt this past workflow to the current requested outcomes.
- 1. list_class_types: supply current values for name. Inspect the result before proceeding.
- 2. list_locations: supply current values for name. Inspect the result before proceeding.
- 3. list_rooms: supply current values for location_id, name. Inspect the result before proceeding.
- 4. list_instructors: supply current values for full_name. Inspect the result before proceeding.
- 5. create_class_session: supply current values for booking_status, class_type_id, end_time, instructor_id, location_id, room_id, start_time. Inspect the result before proceeding.
- Repeat a step only for distinct requested objects. Verify all requested effects; do not assume this example covers extra requirements.

来源：1 条训练轨迹；精确哈希和成员路径见 bank.json。
