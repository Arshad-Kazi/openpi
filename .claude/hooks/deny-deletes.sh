#!/usr/bin/env bash
# PreToolUse(Bash) guard: every command runs freely except ones that delete.
#
# Wired up from .claude/settings.json. Reads the hook payload on stdin and
# scans the whole command string, so it also catches deletions hidden inside
# `find -exec`, `xargs`, `$(...)` and inline python that plain permission
# prefix rules cannot see.
#
#   no output -> command proceeds
#   deny JSON -> command is blocked
#
# Set DECISION to "ask" instead of "deny" to turn hard blocks into a
# confirmation prompt. Trim the pattern groups below to taste.

set -uo pipefail

DECISION="deny"

cmd=$(jq -r '.tool_input.command // empty')
[[ -z $cmd ]] && exit 0

# A deleter only counts when it is in command position: start of a line, after
# a separator, or handed to sudo / xargs / find -exec.
P='(^|[;&|`(]|\$\(|\bsudo[[:space:]]+|\bxargs[[:space:]]+(-[^[:space:]]+[[:space:]]+)*|-exec(dir)?[[:space:]]+|\bthen[[:space:]]+|\bdo[[:space:]]+|\bnohup[[:space:]]+|\btime[[:space:]]+|\benv[[:space:]]+)[[:space:]]*'

patterns=(
  # files and directories
  "${P}(rm|rmdir|unlink|shred|srm|wipe|trash|trash-put|truncate)\b"
  "${P}gio[[:space:]]+trash\b"
  "\bfind\b.*[[:space:]]-delete\b"
  "\brsync\b.*--delete"
  "(shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir|\.unlink\(|\brimraf\b|fs\.rm\b|fs\.unlink)"

  # git history and uncommitted work
  "${P}git[[:space:]]+(clean|rm)\b"
  "\bgit\b[^;&|]*\breset\b[^;&|]*--hard"
  "\bgit\b[^;&|]*\bbranch\b[^;&|]*[[:space:]]-[dD]\b"
  "\bgit\b[^;&|]*\bpush\b[^;&|]*(--force|--delete|[[:space:]]-f\b)"
  "\bgit\b[^;&|]*\bstash\b[^;&|]*[[:space:]](drop|clear)\b"
  "\bgit\b[^;&|]*\b(checkout|restore)\b[^;&|]*[[:space:]]--([[:space:]]|$)"
  "\bgit\b[^;&|]*\b(filter-branch|filter-repo)\b"

  # containers and clusters
  "${P}docker([[:space:]]+(image|container|volume|network|system|builder))?[[:space:]]+(rm|rmi|prune)\b"
  "\bdocker[- ]compose\b[^;&|]*\bdown\b[^;&|]*[[:space:]]-v"
  "\bkubectl[[:space:]]+delete\b"

  # cloud storage
  "\baws\b[^;&|]*\bs3\b[^;&|]*[[:space:]](rm|rb)\b"
  "\bgsutil\b[^;&|]*[[:space:]]rm\b"
  "\bgcloud\b[^;&|]*[[:space:]]delete\b"
  "\baz\b[^;&|]*[[:space:]]delete\b"

  # packages and environments
  "\b(apt|apt-get|aptitude)[[:space:]]+(remove|purge|autoremove)\b"
  "\b(yum|dnf)[[:space:]]+(remove|erase|autoremove)\b"
  "\bpacman[[:space:]]+-R"
  "\bpip3?[[:space:]]+uninstall\b"
  "\bconda[[:space:]]+(env[[:space:]]+remove|remove|uninstall)\b"
  "\bnpm[[:space:]]+(uninstall|rm|prune)\b"

  # disks and devices
  "${P}(mkfs(\.[a-z0-9]+)?|wipefs|blkdiscard|fdisk|parted|sgdisk)\b"
  "\bdd\b[^;&|]*[[:space:]]of="
  ">[[:space:]]*/dev/(sd|nvme|hd|vd|mapper)"
  "\bcrontab[[:space:]]+-r\b"
)

for p in "${patterns[@]}"; do
  if grep -Eq -- "$p" <<<"$cmd"; then
    jq -n --arg d "$DECISION" --arg p "$p" '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: $d,
        permissionDecisionReason:
          "Blocked: this command deletes or destroys data (matched guard pattern \($p)). Deletion is disabled for Claude in this project — run it yourself, or edit .claude/hooks/deny-deletes.sh."
      }
    }'
    exit 0
  fi
done

exit 0