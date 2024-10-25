using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CommentPridictor
{
    public class SentementData
    {
        [LoadColumn(0)]
        public string? sentementText;
        [LoadColumn(1),ColumnName("Label")]
        public bool? sentement;
    }
    public class SentementPridiction : SentementData
    {
        [ColumnName("PridictionLabel")]
        public bool Pridiction { get; set; }
        public float? Probablity { get; set; }
    }
}
