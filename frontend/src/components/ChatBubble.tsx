import React, { useState, useMemo, useEffect } from 'react';
import { ChatMessage } from '../types';
import { FileText, User, Bot, Star, Users, Package, MapPinned, Filter, ChevronDown, BadgePercent, ExternalLink } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatBubbleProps {
  message: ChatMessage;
}

const ChatBubble: React.FC<ChatBubbleProps> = ({ message }) => {
  const isUser = message.type === 'user';
  const hasAttachments = message.attachments && message.attachments.length > 0;
  
  // Filter states
  const [priceSort] = useState<'none' | 'low-high' | 'high-low'>('none');
  const [ratingSort] = useState<'none' | 'high-low' | 'low-high'>('none');
  const [areaFilter, setAreaFilter] = useState<string>('all');
  const [showFilters, setShowFilters] = useState(false);
  const [showAllAmenities, setShowAllAmenities] = useState<Record<number, boolean>>({});
  
  const extractWorkspaceRecommendations = (text: string) => {
    const recommendations: any[] = [];
    const lines = text.split('\n');
    let currentRec: any = {};

    lines.forEach(line => {
      if (line.match(/^\d+\./)) {
        if (Object.keys(currentRec).length > 0) {
          recommendations.push(currentRec);
        }
        // Extract name and area if present in the format: 1. Workspace Name (Area)
        const nameAreaMatch = line.match(/^\d+\.\s+([^()]+?)(?:\s*\(([^)]+)\))?$/);
        if (nameAreaMatch) {
          currentRec = { name: nameAreaMatch[1].trim() };
          if (nameAreaMatch[2]) {
            currentRec.area = nameAreaMatch[2].trim();
          }
        } else {
          currentRec = { name: line.replace(/^\d+\.\s+/, '') };
        }
      } else if (currentRec.name) {
        const addressMatch = line.match(/Address:\s+(.*)/);
        const typeMatch = line.match(/Workspace Type:\s+(.*)/);
        const offeringsMatch = line.match(/Offerings:\s+(.*)/);
        const amenitiesMatch = line.match(/Amenities:\s+(.*)/);
        const seatsMatch = line.match(/Seats Available:\s+(.*)/);
        const ratingMatch = line.match(/Rating:\s+(.*)/);
        const categoryMatch = line.match(/Category:\s+(.*)/);
        const priceMatch = line.match(/Price:\s+₹(.*)/);
        const similarityMatch = line.match(/Similarity Score:\s+([\d.]+)%/);
        const linkMatch = line.match(/Link:\s+\[(.*?)\]\((.*?)\)/);

        if (addressMatch) currentRec.address = addressMatch[1];
        if (typeMatch) currentRec.workspace_type = typeMatch[1];
        if (offeringsMatch) currentRec.offerings = offeringsMatch[1];
        if (amenitiesMatch) currentRec.amenities = amenitiesMatch[1];
        if (seatsMatch) currentRec.seats = seatsMatch[1];
        if (ratingMatch) currentRec.rating = parseFloat(ratingMatch[1]);
        if (categoryMatch) currentRec.category = categoryMatch[1];
        if (priceMatch) currentRec.price = parseInt(priceMatch[1]);
        if (similarityMatch) currentRec.similarity_score = parseFloat(similarityMatch[1]);
        if (linkMatch) {
          currentRec.linkText = linkMatch[1];
          currentRec.linkUrl = linkMatch[2];
        }
      }
    });

    if (Object.keys(currentRec).length > 0) {
      recommendations.push(currentRec);
    }

    return recommendations;
  };

  // Extract Stylework URL from message content
  const extractStyleworkUrl = (text: string): string | null => {
    const urlMatch = text.match(/🔗\s*\*\*Browse more options:\*\*\s*(https?:\/\/[^\s\n]+)/);
    return urlMatch ? urlMatch[1] : null;
  };

  let recommendationsTextToUse = message.content;
  const workspaceRecommendations = !isUser ? extractWorkspaceRecommendations(recommendationsTextToUse) : [];
  const hasRecommendations = workspaceRecommendations.length > 0;
  const styleworkUrl = !isUser ? extractStyleworkUrl(recommendationsTextToUse) : null;

  /*
  // Auto-open Stylework URL when recommendations are shown
  useEffect(() => {
    if (styleworkUrl && hasRecommendations) {
      // Small delay to ensure the message is fully rendered
      const timer = setTimeout(() => {
        console.log(`[DEBUG] Auto-opening Stylework URL: ${styleworkUrl}`);
        window.open(styleworkUrl, '_blank', 'noopener,noreferrer');
      }, 1000);

      return () => clearTimeout(timer);
    }
  }, [styleworkUrl, hasRecommendations]);
  */

  // Area extraction: use only the 'area' field from the workspace object
  const extractAreaFromWorkspace = (workspace: any): string[] => {
    if (workspace.area && typeof workspace.area === 'string' && workspace.area.trim().length > 0) {
      return [workspace.area.trim().toLowerCase()];
    }
    return [];
  };

  // Extract unique areas for the current city from all recommendations (not filtered by similarity score)
  const allAreas = useMemo(() => {
    const areaSet = new Set<string>();
    const allRecs = !isUser ? extractWorkspaceRecommendations(recommendationsTextToUse) : [];
    allRecs.forEach(workspace => {
      const areas = extractAreaFromWorkspace(workspace);
      areas.forEach(area => areaSet.add(area));
    });
    return Array.from(areaSet).sort();
  }, [recommendationsTextToUse, isUser]);

  // Apply filters and sorting
  const filteredAndSortedRecommendations = useMemo(() => {
    let filtered = [...workspaceRecommendations];

    // Apply area filter
    if (areaFilter !== 'all') {
      filtered = filtered.filter(workspace => {
        const areas = extractAreaFromWorkspace(workspace);
        return areas.some(area => area.trim().toLowerCase() === areaFilter.trim().toLowerCase());
      });
    }

    // Only show workspaces with a similarity score (any value, not undefined)
    filtered = filtered.filter(workspace => typeof workspace.similarity_score !== 'undefined' && workspace.similarity_score > 70);

    // Apply sorting
    if (priceSort !== 'none') {
      filtered.sort((a, b) => {
        const priceA = a.price || 0;
        const priceB = b.price || 0;
        return priceSort === 'low-high' ? priceA - priceB : priceB - priceA;
      });
    } else if (ratingSort !== 'none') {
      filtered.sort((a, b) => {
        const ratingA = a.rating || 0;
        const ratingB = b.rating || 0;
        return ratingSort === 'high-low' ? ratingB - ratingA : ratingA - ratingB;
      });
    }

    return filtered;
  }, [workspaceRecommendations, priceSort, ratingSort, areaFilter]);
  
  // Split content to separate intro from recommendations and URL
  const [introText = '', recommendationsText = ''] = (recommendationsTextToUse || '').split('\n\nHere are some workspace recommendations for you:');

  // Check if similarity scores are present in the recommendations text
  const hasSimilarityScores = recommendationsText && recommendationsText.includes('Similarity Score:');

  // Clean and format the intro text for better display
  const cleanIntroText = (text: string): string => {
    if (!text) return '';
    
    // Remove code blocks, JSON objects, and Stylework URL section
    let cleaned = text
      .replace(/```[\s\S]*?```/g, '')
      .replace(/\{[\s\S]*?\}/g, '')
      .replace(/🔗\s*\*\*Browse more options:\*\*\s*https?:\/\/[^\s\n]+/g, '');
    
    // Remove excessive whitespace and empty lines
    cleaned = cleaned.replace(/\n\s*\n\s*\n/g, '\n\n'); // Replace multiple empty lines with double line break
    cleaned = cleaned.replace(/^\s+|\s+$/g, ''); // Trim start and end
    
    return cleaned;
  };

  // Only show introText if it is not empty, not just whitespace, and not just a code block or Gemini JSON
  let showIntro = false;
  let introToShow = '';
  
  if (typeof introText !== 'undefined' && introText.trim().length > 0) {
    const cleanedIntro = cleanIntroText(introText);
    if (cleanedIntro.length > 0) {
      showIntro = true;
      introToShow = cleanedIntro;
    }
  } else if (!hasRecommendations && message.content.trim().length > 0) {
    showIntro = true;
    introToShow = cleanIntroText(message.content);
  }

  const resetFilters = () => {
    setAreaFilter('all');
  }

  const hasActiveFilters = priceSort !== 'none' || ratingSort !== 'none' || areaFilter !== 'all';
  
  // Ensure timestamp is a Date object
  let timestamp: Date;
  if (message.timestamp instanceof Date) {
    timestamp = message.timestamp;
  } else if (typeof message.timestamp === 'string') {
    timestamp = new Date(message.timestamp);
  } else {
    timestamp = new Date();
  }

  // Custom markdown components for better formatting
  const markdownComponents = {
    // Main heading with large bullet
    h1: ({ children }: any) => (
      <h1 className="text-lg font-bold mb-3 mt-4 first:mt-0 flex items-start">
        <span className="inline-block w-2 h-2 bg-blue-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
        <span>{children}</span>
      </h1>
    ),
    
    // Subheading with medium bullet
    h2: ({ children }: any) => (
      <h2 className="text-base font-semibold mb-1 mt-3 first:mt-0 pl-4 relative">
        <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1.5 h-1.5 bg-blue-400 rounded-full"></span>
        <span className="pl-2">{children}</span>
      </h2>
    ),
    
    // Sub-subheading with small bullet
    h3: ({ children }: any) => (
      <h3 className="text-sm font-medium mb-1 mt-2 pl-6 relative">
        <span className="absolute left-3 top-1/2 -translate-y-1/2 w-1 h-1 bg-blue-300 rounded-full"></span>
        <span className="pl-2">{children}</span>
      </h3>
    ),
    
    // Regular paragraphs
    p: ({ children }: any) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
    
    // Unordered lists with circular bullets
    ul: ({ children }: any) => <ul className="mb-3 space-y-1">{children}</ul>,
    
    // List items with custom bullet points
    li: ({ node, ...props }: any) => {
      // Check if this is a direct child of ul (first level)
      const isTopLevel = node?.parent?.tagName === 'ul' || node?.parent?.tagName === 'ol';
      
      return (
        <li className="leading-relaxed flex items-start pl-4">
          <span className="inline-block w-1.5 h-1.5 bg-gray-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
          <div className="flex-1">
            {props.children}
          </div>
        </li>
      );
    },
    
    // Ordered lists with numbers
    ol: ({ children }: any) => <ol className="mb-3 space-y-1">{children}</ol>,
    
    // Bold text
    strong: ({ children }: any) => <strong className="font-semibold text-gray-900">{children}</strong>,
    
    // Links
    a: ({ href, children }: any) => (
      <a href={href} className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    ),
    
    // Blockquotes
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-gray-300 pl-4 py-1 my-2 text-gray-600">
        {children}
      </blockquote>
    ),
    
    // Code blocks
    code: ({ children }: any) => <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">{children}</code>,
    pre: ({ children }: any) => <pre className="bg-gray-100 p-3 rounded-lg overflow-x-auto text-sm font-mono mb-3">{children}</pre>,
    
    // Nested list items
    'li > ul': ({ children }: any) => <ul className="ml-4 mt-1 space-y-1">{children}</ul>,
    'li > ol': ({ children }: any) => <ol className="ml-4 mt-1 space-y-1 list-decimal">{children}</ol>
  };

  return (
    <div className="flex flex-col space-y-4 animate-fadeIn">
      {/* Main message bubble */}
      {showIntro && (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
          <div className={`flex items-start max-w-[80%] md:max-w-[70%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
            <div className={`flex-shrink-0 rounded-full p-2 ${isUser ? 'bg-blue-500 ml-2' : 'bg-teal-600 mr-2'}`}>
              {isUser ? (
                <User size={16} className="text-white" />
              ) : (
                <Bot size={16} className="text-white" />
              )}
            </div>
            <div className={`${
              isUser
                ? 'bg-blue-500 text-white rounded-2xl rounded-tr-none'
                : 'bg-white text-gray-800 rounded-2xl rounded-tl-none border border-gray-100'
            } py-3 px-4 shadow-sm`}>
              <div className="text-sm md:text-base break-words">
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  components={markdownComponents}
                >
                  {introToShow}
                </ReactMarkdown>
              </div>
              {hasAttachments && (
                <div className="mt-2 pt-2 border-t border-opacity-20 border-gray-200">
                  {message.attachments?.map((file, index) => (
                    <div key={index} className="flex items-center text-xs">
                      <FileText size={14} className={isUser ? 'text-blue-100' : 'text-blue-500'} />
                      <span className="ml-1 truncate max-w-[200px]">{file.name}</span>
                    </div>
                  ))}
                </div>
              )}
              <div className={`text-xs mt-2 ${isUser ? 'text-blue-100' : 'text-gray-400'}`}>
                {timestamp.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Filter Controls */}
      {hasRecommendations && workspaceRecommendations.length > 0 && filteredAndSortedRecommendations.length > 0 && (
        <div className="ml-12">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center gap-2 px-3 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium"
              >
                <Filter size={16} className="text-gray-500" />
                Filter by Location
                <ChevronDown 
                  size={16} 
                  className={`text-gray-500 transition-transform ${showFilters ? 'rotate-180' : ''}`} 
                />
              </button>
              {hasActiveFilters && (
                <button
                  onClick={resetFilters}
                  className="px-3 py-2 text-sm text-blue-600 hover:text-blue-800 font-medium"
                >
                  Clear All
                </button>
              )}
            </div>
            <div className="text-sm text-gray-600">
              {filteredAndSortedRecommendations.length} of {workspaceRecommendations.length} workspaces
            </div>
          </div>
          {showFilters && (
            <div className="bg-white border border-gray-200 rounded-lg p-4 mb-4 shadow-sm">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4"> 
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Filter by Location
                  </label>
                  <select
                    value={areaFilter}
                    onChange={(e) => setAreaFilter(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                    disabled={allAreas.length === 0}
                  >
                    <option value="all">All Locations</option>
                    {allAreas.map((area) => (
                      <option key={area} value={area}>
                        {area.charAt(0).toUpperCase() + area.slice(1)}
                      </option>
                    ))}
                  </select>
                  {allAreas.length === 0 && (
                    <p className="text-xs text-gray-500 mt-1">No location information available</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Workspace recommendations cards */}
      {hasRecommendations && workspaceRecommendations.length > 0 && filteredAndSortedRecommendations.length > 0 && (
        <div className="ml-12 overflow-x-auto pb-4">
          <div className="flex gap-4">
            {filteredAndSortedRecommendations.map((workspace, index) => (
              <div
                key={index}
                className="flex-shrink-0 w-72 bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm hover:shadow-md transition-shadow flex flex-col relative"
              >
                {/* Similarity Score Badge: Only show if similarity scores are present and this workspace has one */}
                {hasSimilarityScores && workspace.similarity_score && Number(workspace.similarity_score) > 70 && (
                  <div
                    className="absolute top-3 right-3 bg-gradient-to-r from-green-400 to-blue-500 text-white text-xs font-bold px-3 py-1 rounded-full shadow-md flex items-center z-10 animate-fadeIn"
                    title={`Similarity Score: ${workspace.similarity_score}%`}
                  >
                    <BadgePercent size={14} className="mr-1 text-yellow-300" />
                    {workspace.similarity_score}%
                  </div>
                )}
                
                <div className="flex flex-col h-full p-4">
                  <h3 className="font-semibold text-gray-900 mb-1 pr-16">
                    {workspace.name}
                  </h3>
                  <div className="flex items-start text-sm text-gray-600 mb-3">
                    <MapPinned size={14} className="mr-1 mt-0.5 flex-shrink-0" />
                    <span className="line-clamp-2">{workspace.address}</span>
                  </div>
                  
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center text-sm">
                      <Package size={14} className="mr-2 text-gray-500" />
                      <span className="text-gray-700">{workspace.workspace_type}</span>
                    </div> 
                    <div className="flex items-center text-sm">
                      <Users size={14} className="mr-2 text-gray-500" />
                      <span className="text-gray-700">{workspace.seats} seats</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <Star size={14} className="mr-2 text-yellow-500" />
                      <span className="text-gray-700">{workspace.rating} rating</span>
                    </div>
                  </div>
                  
                  <div className="text-sm text-gray-600 mb-3">
                    <strong>Category:</strong> {workspace.category}
                  </div>
                  
                  <div className="text-sm text-gray-600 mb-3">
                    <strong>Offerings:</strong> {workspace.offerings}
                  </div>
                  
                  {workspace.amenities && (
                    <div className="text-sm text-gray-600 mb-3">
                      <strong>Amenities:</strong>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {(() => {
                          const amenitiesArr = workspace.amenities.split(', ');
                          const isExpanded = !!showAllAmenities[index];
                          const visibleAmenities = isExpanded ? amenitiesArr : amenitiesArr.slice(0, 6);
                          const hiddenCount = amenitiesArr.length - 6;

                          return (
                            <>
                              {visibleAmenities.map((amenity: string, i: number) => (
                                <span
                                  key={i}
                                  className="inline-block bg-gray-100 rounded px-2 py-0.5 text-xs"
                                >
                                  {amenity}
                                </span>
                              ))}
                              {hiddenCount > 0 && !isExpanded && (
                                <button
                                  className="inline-block bg-gray-200 rounded px-2 py-0.5 text-xs text-blue-600 hover:underline"
                                  onClick={() => setShowAllAmenities(prev => ({ ...prev, [index]: true }))}
                                >
                                  +{hiddenCount} more
                                </button>
                              )}
                              {isExpanded && amenitiesArr.length > 6 && (
                                <button
                                  className="inline-block bg-gray-200 rounded px-2 py-0.5 text-xs text-blue-600 hover:underline"
                                  onClick={() => setShowAllAmenities(prev => ({ ...prev, [index]: false }))}
                                >
                                  Show less
                                </button>
                              )}
                            </>
                          );
                        })()}
                      </div>
                    </div>
                  )}
                  
                  <div className="mt-auto flex items-center justify-between pt-4">
                    <div className="text-lg font-semibold text-gray-900">
                      {workspace.workspace_type &&
                        workspace.workspace_type.toLowerCase() === "day pass" ? (
                        <>₹{workspace.price}/day</>
                      ) : (
                        <>₹{workspace.price}/month</>
                      )}
                    </div>
                    <a
                      href={workspace.linkUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-3 py-1.5 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
                    >
                      View Details
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {filteredAndSortedRecommendations.length === 0 && hasActiveFilters && (
            <div className="text-center py-8 text-gray-500">
              <Filter size={48} className="mx-auto mb-4 text-gray-300" />
              <p className="text-lg font-medium mb-2">No workspaces found with this filter</p>
              <p className="text-sm">Try adjusting your filter criteria or clearing all filters.</p>
            </div>
          )}
        </div>
      )}

      {/* Stylework URL Button */}
      {styleworkUrl && hasRecommendations && (
        <div className="ml-12">
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4 mb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <ExternalLink size={20} className="text-blue-600 mr-3" />
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">Browse More Options</h3>
                  <p className="text-sm text-gray-600">Explore additional workspaces on Stylework.city</p>
                </div>
              </div>
              <button
                onClick={() => window.open(styleworkUrl, '_blank', 'noopener,noreferrer')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium text-sm flex items-center gap-2"
              >
                Open Website
                <ExternalLink size={16} />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatBubble;