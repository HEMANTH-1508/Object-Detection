const fs = require("fs");
const path = require("path");


// ============================================================
// CONFIGURATION
// ============================================================
//
// The script, images, and annotation files are all in the
// SAME folder.
//
// Example:
//
// personal_dataset/
// ├── duplicate_annotations.js
// ├── frame_000001.jpg
// ├── frame_000001.txt
// ├── frame_000002.jpg
// ├── frame_000002.txt
// └── ...
//
// ============================================================


// ============================================================
// ANNOTATION DIRECTORY
// ============================================================
//
// __dirname means:
//
// "The folder where this JavaScript file is located."
//
// So we don't need to manually write:
//
// "./annotations"
//
// ============================================================

const ANNOTATION_DIR = __dirname;


// ============================================================
// SOURCE FRAME
// ============================================================
//
// This is the annotation file that contains the annotation
// you want to duplicate.
//
// Example:
//
// SOURCE_FRAME = 1
//
// means:
//
// frame_000001.txt
//
// will be used as the source annotation.
//
// ============================================================

const SOURCE_FRAME = 1;

const START_FRAME = 1;
const END_FRAME = 663;


// ============================================================
// FRAME NUMBER DIGITS
// ============================================================
//
// Your filenames use 6 digits:
//
// 1    → 000001
// 10   → 000010
// 278  → 000278
// 2612 → 002612
//
// ============================================================

const FRAME_DIGITS = 6;


// ============================================================
// CREATE SOURCE FILE NAME
// ============================================================

const sourceFrameNumber = String(
    SOURCE_FRAME
).padStart(
    FRAME_DIGITS,
    "0"
);

const sourceFileName = `frame_${sourceFrameNumber}.json`;


// ============================================================
// CREATE SOURCE FILE PATH
// ============================================================

const sourceFilePath = path.join(
    ANNOTATION_DIR,
    sourceFileName
);


// ============================================================
// CHECK SOURCE FILE
// ============================================================

if (!fs.existsSync(sourceFilePath)) {

    console.error("");

    console.error(
        `Source annotation file not found:\n${sourceFilePath}`
    );

    console.error("");

    process.exit(1);
}


// ============================================================
// READ SOURCE ANNOTATION
// ============================================================
//
// The annotation contents are read once.
//
// Every target frame will receive exactly the same
// annotation contents.
//
// ============================================================

const annotationContent = fs.readFileSync(
    sourceFilePath,
    "utf8"
);


// ============================================================
// DISPLAY INFORMATION
// ============================================================

console.log("");

console.log(
    "=".repeat(60)
);

console.log(
    "DUPLICATING YOLO ANNOTATIONS"
);

console.log(
    "=".repeat(60)
);

console.log("");

console.log(
    `Source annotation: ${sourceFileName}`
);

console.log(
    `Start frame:       ${START_FRAME}`
);

console.log(
    `End frame:         ${END_FRAME}`
);

console.log(
    `Output directory:  ${ANNOTATION_DIR}`
);

console.log("");


// ============================================================
// COPY ANNOTATION
// ============================================================

let copiedCount = 0;


for (
    let frameNumber = START_FRAME;
    frameNumber <= END_FRAME;
    frameNumber++
) {

    // --------------------------------------------------------
    // CONVERT FRAME NUMBER TO 6 DIGITS
    // --------------------------------------------------------
    //
    // Example:
    //
    // 1   → 000001
    // 25  → 000025
    // 278 → 000278
    //
    // --------------------------------------------------------

    const paddedFrameNumber = String(
        frameNumber
    ).padStart(
        FRAME_DIGITS,
        "0"
    );


    // --------------------------------------------------------
    // CREATE TARGET FILE NAME
    // --------------------------------------------------------

    const fileName = `frame_${paddedFrameNumber}.json`;


    // --------------------------------------------------------
    // CREATE TARGET FILE PATH
    // --------------------------------------------------------

    const filePath = path.join(
        ANNOTATION_DIR,
        fileName
    );


    // --------------------------------------------------------
    // WRITE ANNOTATION
    // --------------------------------------------------------

    fs.writeFileSync(
        filePath,
        annotationContent,
        "utf8"
    );


    copiedCount++;


    // --------------------------------------------------------
    // DISPLAY PROGRESS
    // --------------------------------------------------------

    console.log(
        `Created: ${fileName}`
    );
}


// ============================================================
// COMPLETION
// ============================================================

console.log("");

console.log(
    "=".repeat(60)
);

console.log(
    "COMPLETED"
);

console.log(
    "=".repeat(60)
);

console.log("");

console.log(
    `Source annotation: ${sourceFileName}`
);

console.log(
    `Files created:     ${copiedCount}`
);

console.log(
    `Frames:             ${START_FRAME} → ${END_FRAME}`
);

console.log("");

console.log(
    `Output directory:  ${ANNOTATION_DIR}`
);

console.log("");