import hashlib

import numpy as np


class ABSplitter:
    def __init__(self, count_slots, salt_one, salt_two):
        self.count_slots = count_slots
        self.salt_one = salt_one
        self.salt_two = salt_two

        self.slots = np.arange(count_slots)
        self.experiments = []
        self.experiment_to_slots = dict()
        self.slot_to_experiments = dict()

    def split_experiments(self, experiments):
        """Устанавливает множество экспериментов, распределяет их по слотам.

        Нужно определить атрибуты класса:
            self.experiments - список словарей с экспериментами
            self.experiment_to_slots - словарь, {эксперимент: слоты}
            self.slot_to_experiments - словарь, {слот: эксперименты}
        experiments - список словарей, описывающих пилот. Словари содержит три ключа:
            experiment_id - идентификатор пилота,
            count_slots - необходимое кол-во слотов,
            conflict_experiments - list, идентификаторы несовместных экспериментов.
            Пример: {'experiment_id': 'exp_16', 'count_slots': 3, 'conflict_experiments': ['exp_13']}
        return: List[dict], список экспериментов, которые не удалось разместить по слотам.
            Возвращает пустой список, если всем экспериментам хватило слотов.
        """
        self.experiments, pilots_with_no_slot, self.slot_to_experiments, self.experiment_to_slots = (
            self._match_pilot_slot(experiments, self.slots.tolist())
        )
        if len(pilots_with_no_slot) == 0:
            return []
        else:
            return pilots_with_no_slot

    def process_user(self, user_id: str):
        """Определяет в какие эксперименты попадает пользователь.

        Сначала нужно определить слот пользователя.
        Затем для каждого эксперимента в этом слоте выбрать пилотную или контрольную группу.

        user_id - идентификатор пользователя.

        return - (int, List[tuple]), слот и список пар (experiment_id, pilot/control group).
            Example: (2, [('exp 3', 'pilot'), ('exp 5', 'control')]).
        """
        user_slot = self._calc_hash(user_id, self.salt_one, self.count_slots)
        user_groups = []
        for exp_id in self.slot_to_experiments[user_slot]:
            group = 'control' if self._calc_hash(user_id+exp_id, self.salt_two, 2) == 0 else 'pilot'
            user_groups.append((exp_id, group))

        return user_slot, user_groups

    def _calc_hash(self, key: str, salt: str, module: int):
        hash_value = int(hashlib.md5(str.encode(str(key) + str(salt))).hexdigest(), 16)
        return hash_value % module

    def _match_pilot_slot(self, pilots: list, slots: list):
        """Функция распределяет пилоты по слотам.

        pilots: список словарей, описывающих пилот. Содержит ключи:
            pilot_id - идентификатор пилота,
            count_slots - необходимое кол-во слотов,
            conflict_pilots - list, идентификаторы несовместных пилотов.
        slots: список с идентификаторами слотов.

        return: словарь соответствия на каких слотах какие пилоты запускаются,
            {slot_id: list_pilot_id, ...}
        """
        pilots = sorted(pilots, key=lambda x: len(x['conflict_experiments']), reverse=True)

        slot_to_pilot = {slot: [] for slot in slots}
        pilot_to_slot = {pilot['experiment_id']: [] for pilot in pilots}
        pilots_with_no_slot = []
        for pilot in pilots:
            if pilot['count_slots'] > len(slots):
                pilots_with_no_slot.append(pilot)
                continue

            notavailable_slots = []
            for conflict_pilot_id in pilot['conflict_experiments']:
                notavailable_slots += pilot_to_slot[conflict_pilot_id]
            available_slots = list(set(slots) - set(notavailable_slots))

            if pilot['count_slots'] > len(available_slots):
                pilots_with_no_slot.append(pilot)
                continue

            np.random.shuffle(available_slots)
            available_slots_orderby_count_pilot = sorted(
                available_slots,
                key=lambda x: len(slot_to_pilot[x]), reverse=True
            )
            pilot_slots = available_slots_orderby_count_pilot[:pilot['count_slots']]
            pilot_to_slot[pilot['experiment_id']] = pilot_slots
            for slot in pilot_slots:
                slot_to_pilot[slot].append(pilot['experiment_id'])

        return pilots, pilots_with_no_slot, slot_to_pilot, pilot_to_slot
