-- Number the sections of a note automatically: "## Title" -> "1. Title", "### Title" -> "1.1. Title".
-- Numbers already typed in a heading (e.g. "## 2. Title") are removed first, so both styles work.
local h2, h3 = 0, 0

local function strip_number(inlines)
  if #inlines > 0 and inlines[1].t == "Str" and inlines[1].text:match("^%d+[%.%d]*%.?$") then
    table.remove(inlines, 1)
    if #inlines > 0 and inlines[1].t == "Space" then table.remove(inlines, 1) end
  end
  return inlines
end

function Header(el)
  local num
  if el.level == 2 then
    h2, h3 = h2 + 1, 0
    num = h2 .. "."
  elseif el.level == 3 then
    h3 = h3 + 1
    num = h2 .. "." .. h3 .. "."
  else
    return nil
  end
  el.content = strip_number(el.content)
  table.insert(el.content, 1, pandoc.Space())
  table.insert(el.content, 1, pandoc.Span({pandoc.Str(num)}, {class = "secnum"}))
  return el
end
