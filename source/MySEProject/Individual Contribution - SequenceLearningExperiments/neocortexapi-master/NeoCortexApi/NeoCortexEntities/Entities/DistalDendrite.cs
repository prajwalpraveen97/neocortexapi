﻿// Copyright (c) Damir Dobric. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.
using NeoCortexApi.Types;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace NeoCortexApi.Entities
{
    /// <summary>
    /// Implements a distal dendritic segment that is used for learning sequences.
    /// Segments are owned by <see cref="Cell"/>s and in turn own <see cref="Synapse"/>s which are obversely connected to by a "source cell", 
    /// which is the <see cref="Cell"/> which will activate a given <see cref="Synapse"/> owned by this <see cref="Segment"/>.
    /// </summary>
    /// <remarks>
    /// Authors of the JAVA implementation: Chetan Surpur, David Ray
    /// </remarks>
    public class DistalDendrite : Segment, IComparable<DistalDendrite>, IEquatable<DistalDendrite>
    {
        /// <summary>
        /// The cell that owns (parent) the segment.
        /// </summary>        
        public Cell ParentCell ; 

        private long m_LastUsedIteration;

        private int m_Ordinal = -1;

        /// <summary>
        /// the last iteration in which this segment was active.
        /// </summary>
        public long LastUsedIteration { get => m_LastUsedIteration; set => m_LastUsedIteration = value; }

        /// <summary>
        /// The seqence number of the segment. Specifies the order of the segment of the <see cref="Connections"/> instance.
        /// </summary>
        public int Ordinal { get => m_Ordinal; set => m_Ordinal = value; }

        /// <summary>
        /// Default constructor used by deserializer.
        /// </summary>
        public DistalDendrite() 
        { 
        
        }


        /// <summary>
        /// Creates the Distal Segment.
        /// </summary>
        /// <param name="parentCell">The cell, which owns the segment.</param>
        /// <param name="flatIdx">The flat index of the segment. If some segments are destroyed (synapses lost permanence)
        /// then the new segment will reuse the flat index. In contrast, 
        /// the ordinal number will increas when new segments are created.</param>
        /// <param name="lastUsedIteration"></param>
        /// <param name="ordinal">The ordindal number of the segment. This number is incremented on each new segment.
        /// If some segments are destroyed, this number is still incrementd.</param>
        /// <param name="synapsePermConnected"></param>
        /// <param name="numInputs"></param>
        public DistalDendrite(Cell parentCell, int flatIdx, long lastUsedIteration, int ordinal, double synapsePermConnected, int numInputs) : base(flatIdx, synapsePermConnected, numInputs)
        {
            this.ParentCell = parentCell;
            this.m_Ordinal = ordinal;
            this.m_LastUsedIteration = lastUsedIteration;            
        }


        /// <summary>
        /// Gets all synapses owned by this distal dentrite segment.
        /// </summary>
        /// <param name="mem"></param>
        /// <returns>Synapses.</returns>
        public List<Synapse> GetAllSynapses(Connections mem)
        {
            //DD  return mem.GetSynapses(this);
            return this.Synapses;
        }

        /// <summary>
        /// Gets all active synapses of this segment, which have presynaptic cell as active one.
        /// </summary>
        /// <param name="c"></param>
        /// <param name="activeCells"></param>
        /// <returns></returns>
        //public ISet<Synapse> GetActiveSynapses(Connections c, ISet<Cell> activeCells)
        //{
        //    ISet<Synapse> activeSynapses = new LinkedHashSet<Synapse>();

        //    //DD foreach (var synapse in c.GetSynapses(this))
        //    foreach (var synapse in this.Synapses)
        //    {
        //        if (activeCells.Contains(synapse.getPresynapticCell()))
        //        {
        //            activeSynapses.Add(synapse);
        //        }
        //    }

        //    return activeSynapses;
        //}

           
        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return $"DistalDendrite: Indx:{this.SegmentIndex}";
        }


        /// <summary>
        /// <inheritdoc/>
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            int prime = 31;
            int result = base.GetHashCode();
            result = prime * result + ((ParentCell == null) ? 0 : ParentCell.GetHashCode());
            result *= this.SegmentIndex;
            return result;
        }

        /// <summary>
        /// Compares this segment with the given one.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>

        public bool Equals(DistalDendrite obj)
        {
            if (this == obj)
                return true;
            if (obj == null)
                return false;

            DistalDendrite other = (DistalDendrite)obj;
            if (ParentCell == null)
            {
                if (other.ParentCell != null)
                    return false;
            }
            else if (!ParentCell.Equals(other.ParentCell))
                return false;
            if (m_LastUsedIteration != other.m_LastUsedIteration)
                return false;
            if (m_Ordinal != other.m_Ordinal)
                return false;
            if (LastUsedIteration != other.LastUsedIteration)
                return false;
            if (Ordinal != other.Ordinal)
                return false;
            if (SegmentIndex != obj.SegmentIndex)
                return false;
            if (Synapses == null)
            {
                if (obj.Synapses != null)
                    return false;
            }
            else if (!Synapses.SequenceEqual(obj.Synapses))
                return false;
            if (SynapsePermConnected != obj.SynapsePermConnected)
                return false;
            if (NumInputs != obj.NumInputs)
                return false;

            return true;
        }


        /// <summary>
        /// Compares by index.
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public int CompareTo(DistalDendrite other)
        {
            if (this.SegmentIndex > other.SegmentIndex)
                return 1;
            else if (this.SegmentIndex < other.SegmentIndex)
                return -1;
            else
                return 0;
        }

        #region Serialization
        public override void Serialize(StreamWriter writer)
        {
            HtmSerializer2 ser = new HtmSerializer2();

           ser.SerializeBegin(nameof(DistalDendrite), writer);

            if (this.boxedIndex != null)
            {
                this.boxedIndex.Serialize(writer);
            }

            if (this.ParentCell != null)
            {
                // We are serializeing the index of the cell only to avoid circular references during serialization.
                ser.SerializeValue(this.ParentCell.Index, writer);
                //this.ParentCell.Serialize(writer);
            }
            ser.SerializeValue(this.m_LastUsedIteration, writer);
            ser.SerializeValue(this.m_Ordinal, writer);
            ser.SerializeValue(this.LastUsedIteration, writer);
            ser.SerializeValue(this.Ordinal, writer);
            ser.SerializeValue(this.SegmentIndex, writer);
            ser.SerializeValue(this.SynapsePermConnected, writer);
            ser.SerializeValue(this.NumInputs, writer);

            // We serialize synapse indixes only to avoid circular references.
            if (this.Synapses != null && this.Synapses.Count > 0)
                ser.SerializeValue(this.Synapses.Select(s => s.SynapseIndex).ToArray(), writer);
               
            ser.SerializeEnd(nameof(DistalDendrite), writer);
        }


        public static DistalDendrite Deserialize(StreamReader sr)
        {
            DistalDendrite distal = new DistalDendrite();

            HtmSerializer2 ser = new HtmSerializer2();

            while (sr.Peek() >= 0)
            {
                string data = sr.ReadLine();
                if (data == ser.LineDelimiter || data == ser.ReadBegin(nameof(DistalDendrite)))
                {
                    continue;
                }
                else if (data == ser.ReadBegin(nameof(Cell)))
                {
                    //distal.ParentCell = Cell.Deserialize(sr);
                }
                else if (data == ser.ReadBegin(nameof(Integer)))
                {
                    distal.boxedIndex = Integer.Deserialize(sr);
                }
                else if (data == ser.ReadEnd(nameof(DistalDendrite)))
                {
                    break;
                }
                else
                { 
                    string[] str = data.Split(HtmSerializer2.ParameterDelimiter);
                    for (int i = 0; i < str.Length; i++)
                    {
                        switch (i)
                        {
                            case 0:
                                {
                                    distal.ParentCell = new Cell();
                                    distal.ParentCell.Index = ser.ReadIntValue(str[i]);
                                    
                                    break;
                                }
                            case 1:
                                {
                                    distal.m_LastUsedIteration = ser.ReadLongValue(str[i]);
                                    break;
                                }
                            case 2:
                                {
                                    distal.m_Ordinal = ser.ReadIntValue(str[i]);
                                    break;
                                }
                            case 3:
                                {
                                    distal.LastUsedIteration = ser.ReadLongValue(str[i]);
                                    break;
                                }
                            case 4:
                                {
                                    distal.Ordinal = ser.ReadIntValue(str[i]);
                                    break;
                                }
                            case 5:
                                {
                                    distal.SegmentIndex = ser.ReadIntValue(str[i]);
                                    break;
                                }
                            case 6:
                                {
                                    distal.SynapsePermConnected = ser.ReadDoubleValue(str[i]);
                                    break;
                                }
                            case 7:
                                {
                                    distal.NumInputs = ser.ReadIntValue(str[i]);
                                    break;
                                }
                            
                            default:
                                { break; }

                        }
                    }
                }
                
            }

            return distal;

        }

   

        #endregion

    }
}

