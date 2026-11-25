// A Quest 3 trigger reads 0 for the first sixth of its travel, then about the rest of the travel to the power 1.45.
const TRIGGER_INDEX = 0;
const POWER = 1 / 1.45;
// The first profile is the most specific. Other controllers also list `oculus-touch` as a fallback.
const QUEST_PROFILE = /^(meta-quest-touch|oculus-touch-v\d)/;

export function bendQuestTrigger(buttons, profiles) {
  if (!QUEST_PROFILE.test(profiles?.[0] ?? '') || buttons.length <= TRIGGER_INDEX) return buttons;
  const bent = [...buttons];
  bent[TRIGGER_INDEX] = Math.min(Math.max(buttons[TRIGGER_INDEX], 0), 1) ** POWER;
  return bent;
}
