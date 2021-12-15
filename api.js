const express = require('express');
const fs = require('fs');
const app = express();

app.post('/saveClicks', express.json(), (req, res)=>{
    const clicks = req.body?.clicks || 0;
    const total = req.body.total

    fs.writeFileSync(`clicks/${total}_${Date.now()}.txt`, clicks);
    res.status(200).json({
        success: true
    })
})

app.listen(3000);