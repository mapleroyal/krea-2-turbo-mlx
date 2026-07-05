export function fileDragStateFromTransfer(dataTransfer, acceptsFileType) {
  if (!hasDraggedFiles(dataTransfer)) {
    return "idle";
  }

  return canAcceptSingleDraggedFile(dataTransfer.items, acceptsFileType)
    ? "accept"
    : "reject";
}

export function hasDraggedFiles(dataTransfer) {
  const types = Array.from(dataTransfer?.types ?? []);
  return (
    types.includes("Files") ||
    Array.from(dataTransfer?.items ?? []).some((item) => item.kind === "file")
  );
}

function canAcceptSingleDraggedFile(items, acceptsFileType) {
  const draggedItems = Array.from(items ?? []);
  if (draggedItems.length === 0) {
    return true;
  }

  const fileItems = draggedItems.filter((item) => item.kind === "file");
  if (fileItems.length !== 1) {
    return false;
  }

  const type = String(fileItems[0].type ?? "").toLowerCase();
  return !type || acceptsFileType(type);
}
