-- Remove the first "# Title" from a note: the page template shows it above the contents box.
local done = false
function Header(el)
  if not done and el.level == 1 then
    done = true
    return {}
  end
end
